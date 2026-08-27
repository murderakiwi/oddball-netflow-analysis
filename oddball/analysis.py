"""Descriptive and temporal analysis."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

from .data import WEIGHT_COLS, WEIGHT_LABELS, build_edge_table, positive_edge_medians
from .features import Graph, egonet_features
from .scoring import DEFAULT_K_RANGE, fit_r2, jaccard, score_features

PAIRS = [("n_flows", "n_packets"), ("n_flows", "n_bytes"), ("n_packets", "n_bytes")]
PAIR_LABELS = {
    ("n_flows", "n_packets"): "flow-packet",
    ("n_flows", "n_bytes"): "flow-byte",
    ("n_packets", "n_bytes"): "packet-byte",
}

COMPOSITE_WEIGHTS = {
    "geo_flow_packet_25_75": (0.25, 0.75, 0.00),
    "geo_flow_packet_50_50": (0.50, 0.50, 0.00),
    "geo_flow_packet_75_25": (0.75, 0.25, 0.00),
    "geo_flow_byte_25_75": (0.25, 0.00, 0.75),
    "geo_flow_byte_50_50": (0.50, 0.00, 0.50),
    "geo_flow_byte_75_25": (0.75, 0.00, 0.25),
    "geo_packet_byte_25_75": (0.00, 0.25, 0.75),
    "geo_packet_byte_50_50": (0.00, 0.50, 0.50),
    "geo_packet_byte_75_25": (0.00, 0.75, 0.25),
    "geo_equal_triplet": (1 / 3, 1 / 3, 1 / 3),
    "geo_50_25_25": (0.50, 0.25, 0.25),
    "geo_60_20_20": (0.60, 0.20, 0.20),
    "geo_80_10_10": (0.80, 0.10, 0.10),
    "geo_20_50_30": (0.20, 0.50, 0.30),
    "geo_20_30_50": (0.20, 0.30, 0.50),
}
COMPOSITE_LABELS = {
    "geo_flow_packet_25_75": "Geometric Flow/Packet 25/75",
    "geo_flow_packet_50_50": "Geometric Flow/Packet 50/50",
    "geo_flow_packet_75_25": "Geometric Flow/Packet 75/25",
    "geo_flow_byte_25_75": "Geometric Flow/Byte 25/75",
    "geo_flow_byte_50_50": "Geometric Flow/Byte 50/50",
    "geo_flow_byte_75_25": "Geometric Flow/Byte 75/25",
    "geo_packet_byte_25_75": "Geometric Packet/Byte 25/75",
    "geo_packet_byte_50_50": "Geometric Packet/Byte 50/50",
    "geo_packet_byte_75_25": "Geometric Packet/Byte 75/25",
    "geo_equal_triplet": "Geometric Equal Triplet",
    "geo_50_25_25": "Geometric 50/25/25",
    "geo_60_20_20": "Geometric 60/20/20",
    "geo_80_10_10": "Geometric 80/10/10",
    "geo_20_50_30": "Geometric 20/50/30",
    "geo_20_30_50": "Geometric 20/30/50",
}

_PARALLEL_EDGES = None
_PARALLEL_K = DEFAULT_K_RANGE


def _init_parallel_scoring(edges, k):
    global _PARALLEL_EDGES, _PARALLEL_K
    _PARALLEL_EDGES = edges
    _PARALLEL_K = k


def _score_ewpl_column(column):
    graph = Graph(_PARALLEL_EDGES, column)
    features = egonet_features(graph, include_eigenvalue=False)
    return column, score_features(features, k=_PARALLEL_K, laws=("EWPL",))


def score_all_weightings(
    edges: pd.DataFrame, weight_cols=None, k=DEFAULT_K_RANGE, include_eigenvalue=False
):
    weight_cols = list(weight_cols or WEIGHT_COLS)
    out = {}
    shared_edpl = None
    shared_attrs = {}
    for position, c in enumerate(weight_cols):
        g = Graph(edges, c)
        feat = egonet_features(g, include_eigenvalue=include_eigenvalue)
        laws = ["EWPL"]
        if include_eigenvalue:
            laws.append("ELWPL")
        if position == 0:
            laws.append("EDPL")
        frame = score_features(feat, k=k, laws=laws)
        if position == 0:
            shared_edpl = frame[
                [col for col in frame if col.startswith("edpl_")]
            ].copy()
            shared_attrs = {
                key: value
                for key, value in frame.attrs.items()
                if key.startswith("edpl_")
            }
        else:
            if not frame["node"].equals(out[weight_cols[0]]["node"]):
                raise ValueError("node order differs between weightings")
            for col in shared_edpl:
                frame[col] = shared_edpl[col].to_numpy()
            frame.attrs.update(shared_attrs)
        out[c] = frame
    return out


def score_ewpl_weightings_parallel(
    edges: pd.DataFrame, weight_cols, k=DEFAULT_K_RANGE, processes=4
):
    columns = list(weight_cols)
    if processes <= 1:
        _init_parallel_scoring(edges, k)
        return dict(_score_ewpl_column(column) for column in columns)
    context = (
        mp.get_context("fork")
        if "fork" in mp.get_all_start_methods()
        else mp.get_context()
    )
    with context.Pool(
        processes=min(processes, len(columns)),
        initializer=_init_parallel_scoring,
        initargs=(edges, k),
    ) as pool:
        return dict(pool.map(_score_ewpl_column, columns, chunksize=1))


def geometric_composite_edges(edges: pd.DataFrame, definitions):
    out = edges.copy()
    medians = positive_edge_medians(edges)
    log_scaled = [
        np.log(edges[column].to_numpy(float) / medians[column])
        for column in WEIGHT_COLS
    ]
    for method, coefficients in definitions.items():
        out[method] = np.exp(
            sum(
                coefficient * values
                for coefficient, values in zip(coefficients, log_scaled)
            )
        )
    return out


def composite_ewpl_summary(scored, labels, top_k=50):
    rows = []
    for method, frame in scored.items():
        r2, _, bins = fit_r2(frame["E"].to_numpy(float), frame["W"].to_numpy(float))
        top = frame.nlargest(top_k, "ewpl_score")
        rows.append(
            {
                "method": method,
                "label": labels.get(method, method),
                "C": frame.attrs["ewpl_C"],
                "theta": frame.attrs["ewpl_theta"],
                "binned_median_r2": r2,
                "fit_bins": bins,
                f"top{top_k}_heavy": int((top["ewpl_side"] == "heavy").sum()),
                f"top{top_k}_light": int((top["ewpl_side"] == "light").sum()),
                f"top{top_k}_on_line": int((top["ewpl_side"] == "on-line").sum()),
                f"top{top_k}_lof_driven": int(
                    top["ewpl_score_driver"].str.startswith("LOF").sum()
                ),
                f"top{top_k}_line_driven": int(
                    (top["ewpl_score_driver"] == "line-dominant").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _law_prefix(law: str) -> str:
    prefix = law.lower()
    if prefix not in {"edpl", "ewpl", "elwpl"}:
        raise ValueError("law must be EDPL, EWPL, or ELWPL")
    return prefix


def cross_weight_overlap(scored: dict, law: str = "EWPL", K: int = 20) -> pd.DataFrame:
    prefix = _law_prefix(law)
    col = f"{prefix}_score"
    tops = {
        weight: set(frame.nlargest(K, col)["node"]) for weight, frame in scored.items()
    }
    rows = []
    for a, b in PAIRS:
        if a in tops and b in tops:
            rows.append(
                {
                    "pair": PAIR_LABELS[(a, b)],
                    "K": K,
                    "jaccard": jaccard(tops[a], tops[b]),
                }
            )
    return pd.DataFrame(rows)


def k_curve(
    scored: dict, law: str = "EWPL", Ks=(5, 10, 20, 30, 50, 75, 100)
) -> pd.DataFrame:
    frames = [cross_weight_overlap(scored, law=law, K=K) for K in Ks]
    return pd.concat(frames, ignore_index=True)


def rank_correlation(
    scored: dict, law: str = "EWPL", method: str = "spearman"
) -> pd.DataFrame:
    col = f"{_law_prefix(law)}_score"
    nodes = None
    series = {}
    for c, df in scored.items():
        d = df.set_index("node")[col]
        series[c] = d
        nodes = d.index if nodes is None else nodes.intersection(d.index)
    M = pd.DataFrame({c: series[c].loc[nodes] for c in series})
    corr = M.corr(method=method)
    corr.index = [WEIGHT_LABELS.get(i, i) for i in corr.index]
    corr.columns = [WEIGHT_LABELS.get(c, c) for c in corr.columns]
    return corr


def temporal_replication(
    windows,
    has_header=False,
    k_lof=DEFAULT_K_RANGE,
    laws=("EDPL", "EWPL", "ELWPL"),
    verbose=True,
):
    from .data import load_window

    if isinstance(laws, str):
        laws = (laws,)
    laws = tuple(law.upper() for law in laws)
    for law in laws:
        _law_prefix(law)
    include_eigenvalue = "ELWPL" in laws
    fit_rows = []
    for w in windows:
        if verbose:
            print(f"  window {w['name']} ...", flush=True)
        flows = load_window(
            w["file"],
            w["start"],
            w["end"],
            has_header=has_header,
            relative=w.get("relative", True),
        )
        edges = build_edge_table(flows)
        scored = score_all_weightings(
            edges, k=k_lof, include_eigenvalue=include_eigenvalue
        )
        for law in laws:
            prefix = _law_prefix(law)
            xcol, ycol = {
                "edpl": ("N", "E"),
                "ewpl": ("E", "W"),
                "elwpl": ("W", "lambda_w"),
            }[prefix]
            for weight, frame in scored.items():
                if law == "EDPL" and weight != "n_flows":
                    continue
                r2, _, n_bins = fit_r2(
                    frame[xcol].to_numpy(float), frame[ycol].to_numpy(float)
                )
                fit_rows.append(
                    {
                        "window": w["name"],
                        "law": law,
                        "weight": "-"
                        if law == "EDPL"
                        else WEIGHT_LABELS.get(weight, weight),
                        "theta": frame.attrs[f"{prefix}_theta"],
                        "C": frame.attrs[f"{prefix}_C"],
                        "binned_median_r2": r2,
                        "n_log_bins": n_bins,
                    }
                )
    return pd.DataFrame(fit_rows)


def summarise_temporal(fits: pd.DataFrame) -> pd.DataFrame:
    return (
        fits.groupby(["law", "weight"])
        .agg(
            n_windows=("window", "nunique"),
            theta_mean=("theta", "mean"),
            theta_sd=("theta", "std"),
            theta_min=("theta", "min"),
            theta_max=("theta", "max"),
            r2_mean=("binned_median_r2", "mean"),
            r2_sd=("binned_median_r2", "std"),
            r2_min=("binned_median_r2", "min"),
            r2_max=("binned_median_r2", "max"),
        )
        .reset_index()
    )
