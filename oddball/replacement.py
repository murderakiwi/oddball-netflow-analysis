"""Controlled egonet activity replacement experiment."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from .analysis import COMPOSITE_LABELS, COMPOSITE_WEIGHTS
from .data import build_edge_table, load_window, positive_edge_medians
from .features import Graph, egonet_features
from .scoring import score_law

RAW_COLUMNS = ("n_flows", "n_packets", "n_bytes")
SIMPLE_METHODS = ("flow", "packets", "bytes")
METHODS = SIMPLE_METHODS + tuple(COMPOSITE_WEIGHTS)
CUTOFFS = (10, 20, 50)
DISPLAY = {
    "flow": "Flow",
    "packets": "Packets",
    "bytes": "Bytes",
    **COMPOSITE_LABELS,
}

G_RAW: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
G_INCIDENCE: csr_matrix | None = None
G_E: np.ndarray | None = None
G_CLEAN: dict | None = None
G_K_RANGE = (10, 50)


def _init_worker(raw, incidence, E, clean, k_range):
    global G_RAW, G_INCIDENCE, G_E, G_CLEAN, G_K_RANGE
    G_RAW = raw
    G_INCIDENCE = incidence
    G_E = E
    G_CLEAN = clean
    G_K_RANGE = k_range


def _incidence_matrix(graph: Graph) -> csr_matrix:
    edge_ids = np.arange(len(graph.ei), dtype=np.int64)
    rows = [graph.ei.copy(), graph.ej.copy()]
    cols = [edge_ids, edge_ids]
    for edge_id, (u, v) in enumerate(zip(graph.ei, graph.ej)):
        common = graph.adj[u] & graph.adj[v]
        if common:
            indices = np.fromiter(common, dtype=np.int64, count=len(common))
            rows.append(indices)
            cols.append(np.full(len(indices), edge_id, dtype=np.int64))
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    values = np.ones(len(row), dtype=float)
    return coo_matrix((values, (row, col)), shape=(graph.n, len(graph.ei))).tocsr()


def _rank_scores(scores):
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks, order


def _edge_weights(raw, medians):
    scaled = [raw[i] / medians[RAW_COLUMNS[i]] for i in range(3)]
    logs = [np.log(np.maximum(values, 1e-12)) for values in scaled]
    weights = {"flow": raw[0], "packets": raw[1], "bytes": raw[2]}
    for name, coefficients in COMPOSITE_WEIGHTS.items():
        weights[name] = np.exp(
            sum(coefficient * values for coefficient, values in zip(coefficients, logs))
        )
    return weights


def _positive_medians(raw):
    return {
        name: float(np.median(values[values > 0]))
        for name, values in zip(RAW_COLUMNS, raw)
    }


def _day_path(data_dir: Path, day: int) -> Path:
    return data_dir / f"netflow_day-{day:02d}.bz2"


def _load_graph(path: Path, start: float, end: float):
    flows = load_window(path, start, end, relative=True)
    edges = build_edge_table(flows)
    graph = Graph(edges, "n_flows")
    base = egonet_features(graph)
    incidence = _incidence_matrix(graph)
    raw = tuple(edges[column].to_numpy(float) for column in RAW_COLUMNS)
    totals = np.column_stack([np.asarray(incidence @ values).ravel() for values in raw])
    return flows, edges, graph, base, incidence, raw, totals


def _history_day(day: int, args):
    started = perf_counter()
    path = _day_path(Path(args.data_dir), day)
    flows, edges, graph, base, incidence, raw, totals = _load_graph(
        path, args.start, args.end
    )
    E = base["E"].to_numpy(float)
    profile = pd.DataFrame(
        {
            "day": day,
            "node": graph.nodes.astype(str),
            "N": base["N"].to_numpy(float),
            "E": E,
            "egonet_flow_total": totals[:, 0],
            "egonet_packet_total": totals[:, 1],
            "egonet_byte_total": totals[:, 2],
        }
    )
    weights = _edge_weights(raw, positive_edge_medians(edges))
    normal_mask = np.ones(graph.n, dtype=bool)
    for method in METHODS:
        W = np.asarray(incidence @ weights[method]).ravel()
        scores = score_law(E, W, k=(args.lof_k_min, args.lof_k_max))["score"]
        _, order = _rank_scores(scores)
        normal_mask[order[: args.history_top_k]] = False

    normal = set(graph.nodes[normal_mask].astype(str))
    print(
        f"History day {day:02d}: {len(flows):,} flows, {len(edges):,} edges, "
        f"{graph.n:,} nodes ({perf_counter() - started:.1f}s)",
        flush=True,
    )
    return profile, normal


def _history_task(task):
    day, args = task
    return day, _history_day(day, args)


def _historical_pool(args, outdir: Path):
    days = list(range(args.history_start_day, args.history_end_day + 1))
    tasks = [(day, args) for day in days]
    if args.history_processes > 1:
        context = (
            mp.get_context("fork")
            if "fork" in mp.get_all_start_methods()
            else mp.get_context()
        )
        with context.Pool(processes=args.history_processes) as pool:
            day_results = list(pool.imap(_history_task, tasks, chunksize=1))
    else:
        day_results = [_history_task(task) for task in tasks]

    eligible = None
    profiles = []
    for _, (profile, normal) in day_results:
        profile["node"] = profile["node"].astype(str)
        eligible = normal if eligible is None else eligible & normal
        profiles.append(profile)
    if not eligible:
        raise RuntimeError("the historical host intersection is empty")

    history = pd.concat(
        [profile[profile["node"].isin(eligible)] for profile in profiles],
        ignore_index=True,
    )
    typical = history.groupby("node", as_index=False).agg(
        history_days=("day", "nunique"),
        median_egonet_flow=("egonet_flow_total", "median"),
        median_egonet_packets=("egonet_packet_total", "median"),
        median_egonet_bytes=("egonet_byte_total", "median"),
    )
    activity = [
        "median_egonet_flow",
        "median_egonet_packets",
        "median_egonet_bytes",
    ]
    q25 = typical[activity].quantile(0.25)
    q75 = typical[activity].quantile(0.75)
    typical["donor_eligible"] = typical[activity].gt(q75, axis="columns").all(axis=1)
    typical["recipient_eligible"] = (
        typical[activity].lt(q25, axis="columns").all(axis=1)
    )
    typical.to_csv(outdir / "historical_host_pool.csv", index=False)

    quantiles = pd.DataFrame({"q25": q25, "q75": q75})
    quantiles.index.name = "historical_host_median_egonet_activity"
    quantiles.to_csv(outdir / "historical_activity_thresholds.csv")
    return typical, quantiles


def _largest_remainder(total: int, shares: np.ndarray, lower: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=np.int64)
    if total < int(lower.sum()):
        raise ValueError("total smaller than required lower bounds")
    shares = np.asarray(shares, float)
    shares = shares / shares.sum()
    target = shares * int(total - lower.sum())
    allocation = lower + np.floor(target).astype(np.int64)
    residual = total - int(allocation.sum())
    if residual:
        order = np.argsort(-(target - np.floor(target)), kind="stable")
        allocation[order[:residual]] += 1
    return allocation


def _integer_allocations(donor_totals, clean_flow):
    totals = np.rint(donor_totals).astype(np.int64)
    edge_count = len(clean_flow)
    if totals[0] < edge_count:
        raise ValueError("donor flow total is smaller than recipient degree")
    shares = np.asarray(clean_flow, float)
    shares = shares / shares.sum()
    flow = _largest_remainder(totals[0], shares, np.ones(edge_count, dtype=np.int64))
    if not (totals[1] >= totals[0] and totals[2] >= totals[1]):
        raise ValueError("donor totals do not satisfy flows <= packets <= bytes")
    packets = _largest_remainder(totals[1], shares, flow)
    bytes_ = _largest_remainder(totals[2], shares, packets)
    return flow, packets, bytes_


def _make_jobs(graph, base, incidence, raw, totals, historical, args):
    node_to_index = {str(node): index for index, node in enumerate(graph.nodes)}
    present = historical[historical["node"].isin(node_to_index)].copy()
    donors = np.asarray(
        [
            node_to_index[node]
            for node in present.loc[present["donor_eligible"], "node"]
        ],
        dtype=int,
    )
    recipients = np.asarray(
        [
            node_to_index[node]
            for node in present.loc[present["recipient_eligible"], "node"]
        ],
        dtype=int,
    )
    if len(donors) < args.trials or len(recipients) < args.trials:
        raise RuntimeError("insufficient eligible donors or recipients")

    rng = np.random.default_rng(args.seed)
    donor_order = list(map(int, rng.permutation(donors)))
    used = set()
    jobs = []
    for recipient in map(int, rng.permutation(recipients)):
        if recipient in used:
            continue
        edge_ids = incidence.getrow(recipient).indices.astype(np.int64, copy=False)
        affected = np.unique(incidence[:, edge_ids].nonzero()[0])
        selected = None
        for donor in map(int, rng.permutation(donor_order)):
            if donor in used or donor in affected:
                continue
            if not np.all(totals[donor] > totals[recipient]):
                continue
            try:
                flow, packets, bytes_ = _integer_allocations(
                    totals[donor], raw[0][edge_ids]
                )
            except ValueError:
                continue
            selected = donor, flow, packets, bytes_
            break
        if selected is None:
            continue
        donor, flow, packets, bytes_ = selected
        used.update((recipient, donor))
        jobs.append(
            {
                "trial_id": len(jobs) + 1,
                "recipient_index": recipient,
                "donor_index": donor,
                "recipient_node": str(graph.nodes[recipient]),
                "donor_node": str(graph.nodes[donor]),
                "recipient_N": float(base.iloc[recipient]["N"]),
                "recipient_E": float(base.iloc[recipient]["E"]),
                "donor_N": float(base.iloc[donor]["N"]),
                "donor_E": float(base.iloc[donor]["E"]),
                "recipient_egonet_totals": totals[recipient].tolist(),
                "donor_egonet_totals": totals[donor].tolist(),
                "egonet_edges": edge_ids.tolist(),
                "affected_indices": affected.tolist(),
                "flow_allocation": flow.tolist(),
                "packet_allocation": packets.tolist(),
                "byte_allocation": bytes_.tolist(),
            }
        )
        if len(jobs) == args.trials:
            break
    if len(jobs) != args.trials:
        raise RuntimeError(f"could construct only {len(jobs)} of {args.trials} pairs")
    return jobs, {
        "test_day_historical_pool_present": len(present),
        "test_day_donor_candidates": len(donors),
        "test_day_recipient_candidates": len(recipients),
    }


def _trial_worker(job):
    recipient = int(job["recipient_index"])
    edge_ids = np.asarray(job["egonet_edges"], dtype=np.int64)
    allocations = [
        np.asarray(job["flow_allocation"], dtype=float),
        np.asarray(job["packet_allocation"], dtype=float),
        np.asarray(job["byte_allocation"], dtype=float),
    ]
    raw = tuple(values.copy() for values in G_RAW)
    for axis in range(3):
        raw[axis][edge_ids] = allocations[axis]
    weights = _edge_weights(raw, _positive_medians(raw))
    rows = []
    for method in job.get("methods", METHODS):
        W = np.asarray(G_INCIDENCE @ weights[method]).ravel()
        scores = score_law(G_E, W, k=G_K_RANGE)["score"]
        ranks, _ = _rank_scores(scores)
        rows.append(
            {
                "trial_id": int(job["trial_id"]),
                "method": method,
                "recipient_node": job["recipient_node"],
                "donor_node": job["donor_node"],
                "clean_rank": int(G_CLEAN[method][recipient]),
                "post_rank": int(ranks[recipient]),
            }
        )
    return rows


def _pair_frame(jobs):
    list_fields = {
        "recipient_egonet_totals",
        "donor_egonet_totals",
        "egonet_edges",
        "affected_indices",
        "flow_allocation",
        "packet_allocation",
        "byte_allocation",
    }
    rows = []
    for job in jobs:
        row = {
            key: json.dumps(value) if key in list_fields else value
            for key, value in job.items()
        }
        for prefix in ("recipient", "donor"):
            totals = job[f"{prefix}_egonet_totals"]
            row[f"{prefix}_egonet_flow"] = totals[0]
            row[f"{prefix}_egonet_packets"] = totals[1]
            row[f"{prefix}_egonet_bytes"] = totals[2]
        rows.append(row)
    return pd.DataFrame(rows)


def _clean_recipient_status(jobs, clean):
    rows = []
    for job in jobs:
        index = int(job["recipient_index"])
        for method in METHODS:
            rank = int(clean[method][index])
            rows.append(
                {
                    "trial_id": int(job["trial_id"]),
                    "node": job["recipient_node"],
                    "method": method,
                    "label": DISPLAY[method],
                    "clean_rank": rank,
                    "clean_top10": rank <= 10,
                    "clean_top20": rank <= 20,
                    "clean_top50": rank <= 50,
                }
            )
    return pd.DataFrame(rows)


def _cutoff_sensitivity(results):
    rows = []
    for cutoff in CUTOFFS:
        for method in METHODS:
            group = results[results["method"] == method]
            detections = int((group["post_rank"] <= cutoff).sum())
            rows.append(
                {
                    "cutoff": cutoff,
                    "method": method,
                    "label": DISPLAY[method],
                    "trials": len(group),
                    "post_alerts": detections,
                    "post_rate": detections / len(group),
                }
            )
    return pd.DataFrame(rows)


def _boolean_frame(results, cutoff):
    wide = results.pivot(index="trial_id", columns="method", values="post_rank")
    if wide.isna().any().any():
        raise ValueError("paired comparison contains missing outcomes")
    return wide.le(cutoff)


def _paired(results, cutoff, baseline="flow"):
    detected = _boolean_frame(results, cutoff)
    reference = detected[baseline]
    rows = []
    for method in COMPOSITE_WEIGHTS:
        composite = detected[method]
        rows.append(
            {
                "cutoff": cutoff,
                "method": method,
                "label": DISPLAY[method],
                "baseline": baseline,
                "both": int((composite & reference).sum()),
                "composite_only": int((composite & ~reference).sum()),
                "baseline_only": int((~composite & reference).sum()),
                "neither": int((~composite & ~reference).sum()),
                "composite_rate": composite.mean(),
                "baseline_rate": reference.mean(),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="..")
    parser.add_argument("--history-start-day", type=int, default=3)
    parser.add_argument("--history-end-day", type=int, default=32)
    parser.add_argument("--test-day", type=int, default=33)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=3600.0)
    parser.add_argument("--history-top-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lof-k-min", type=int, default=10)
    parser.add_argument("--lof-k-max", type=int, default=50)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--history-processes", type=int, default=1)
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument(
        "--outdir",
        default="results/replacement/hour00",
    )
    return parser.parse_args()


def main():
    global G_RAW, G_INCIDENCE, G_E, G_CLEAN, G_K_RANGE
    args = parse_args()
    if args.top_k != max(CUTOFFS):
        raise ValueError("top-k must equal the largest reported alert budget")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    historical, quantiles = _historical_pool(args, outdir)
    if args.history_only:
        return

    path = _day_path(Path(args.data_dir), args.test_day)
    flows, edges, graph, base, incidence, raw, totals = _load_graph(
        path, args.start, args.end
    )
    G_RAW, G_INCIDENCE = raw, incidence
    G_E = base["E"].to_numpy(float)
    G_K_RANGE = (args.lof_k_min, args.lof_k_max)

    clean_weights = _edge_weights(raw, positive_edge_medians(edges))
    G_CLEAN = {}
    for method in METHODS:
        W = np.asarray(incidence @ clean_weights[method]).ravel()
        scores = score_law(G_E, W, k=G_K_RANGE)["score"]
        G_CLEAN[method], _ = _rank_scores(scores)

    jobs, pool_meta = _make_jobs(graph, base, incidence, raw, totals, historical, args)
    pool_meta["persistent_pool"] = len(historical)
    _pair_frame(jobs).to_csv(outdir / "trial_pairs.csv", index=False)

    context = (
        mp.get_context("fork")
        if "fork" in mp.get_all_start_methods()
        else mp.get_context()
    )
    trial_rows = []
    with context.Pool(
        processes=args.processes,
        initializer=_init_worker,
        initargs=(raw, incidence, G_E, G_CLEAN, G_K_RANGE),
    ) as pool:
        for rows in pool.imap_unordered(_trial_worker, jobs, chunksize=1):
            trial_rows.extend(rows)

    results = pd.DataFrame(trial_rows).sort_values(["trial_id", "method"])[
        [
            "trial_id",
            "method",
            "recipient_node",
            "donor_node",
            "clean_rank",
            "post_rank",
        ]
    ]
    raw_path = outdir / "trial_method_results.csv"
    results.to_csv(raw_path, index=False)

    _cutoff_sensitivity(results).to_csv(outdir / "cutoff_sensitivity.csv", index=False)
    for cutoff in CUTOFFS:
        _paired(results, cutoff).to_csv(
            outdir / f"paired_top{cutoff}_vs_flow.csv", index=False
        )
    _clean_recipient_status(jobs, G_CLEAN).to_csv(
        outdir / "selected_host_clean_rank_status.csv", index=False
    )

    metadata = {
        **pool_meta,
        "history_days": list(range(args.history_start_day, args.history_end_day + 1)),
        "test_day": args.test_day,
        "window": [args.start, args.end],
        "historical_screen_top_k": args.history_top_k,
        "methods": list(METHODS),
        "composite_weights": COMPOSITE_WEIGHTS,
        "trials": args.trials,
        "seed": args.seed,
        "alert_budgets": list(CUTOFFS),
        "lof_k_range": list(G_K_RANGE),
        "held_out_graph": {
            "flows": len(flows),
            "edges": len(edges),
            "nodes": graph.n,
        },
        "historical_activity_thresholds": quantiles.to_dict(),
    }
    (outdir / "settings.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Finished in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
