"""OddBall and LOF scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_K_RANGE = (10, 50)


def log_bin_medians(x: np.ndarray, y: np.ndarray, base: float = 2.0):
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return x, y
    lo = np.floor(np.log(x.min()) / np.log(base))
    hi = np.ceil(np.log(x.max()) / np.log(base)) + 1
    edges = base ** np.arange(lo, hi + 1)
    idx = np.digitize(x, edges) - 1
    bx, by = [], []
    for b in np.unique(idx):
        sel = idx == b
        bx.append(np.median(x[sel]))
        by.append(np.median(y[sel]))
    return np.asarray(bx), np.asarray(by)


def fit_power_law(
    x: np.ndarray, y: np.ndarray, on_medians: bool = True, base: float = 2.0
):
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return 1.0, 1.0
    if on_medians:
        bx, by = log_bin_medians(x, y, base=base)
        if len(bx) < 3:
            bx, by = x[m], y[m]
    else:
        bx, by = x[m], y[m]
    theta, logC = np.polyfit(np.log(bx), np.log(by), 1)
    return float(np.exp(logC)), float(theta)


def fit_r2(x: np.ndarray, y: np.ndarray, on_medians: bool = True, base: float = 2.0):
    C, theta = fit_power_law(x, y, on_medians=on_medians, base=base)
    if on_medians:
        bx, by = log_bin_medians(x, y, base=base)
    else:
        m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        bx, by = x[m], y[m]
    ly = np.log(by)
    pred = np.log(C) + theta * np.log(bx)
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), theta, len(bx)


def out_of_line(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    hi = np.maximum(y, yhat)
    lo = np.minimum(y, yhat)
    lo = np.where(lo <= 0, 1e-12, lo)
    return (hi / lo) * np.log1p(np.abs(y - yhat))


def _normalise_k_range(k) -> tuple[int, int]:
    if isinstance(k, (int, np.integer)):
        lower = upper = int(k)
    else:
        try:
            lower, upper = (int(v) for v in k)
        except (TypeError, ValueError):
            raise ValueError("k must be a positive integer or a (lower, upper) pair")
    if lower < 1 or upper < lower:
        raise ValueError("k range must satisfy 1 <= lower <= upper")
    return lower, upper


def lof_k_distinct_range(
    coords: np.ndarray, k_min: int = 10, k_max: int = 50, return_k: bool = False
):
    k_min, k_max = _normalise_k_range((k_min, k_max))
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise ValueError("coords must be a two-dimensional array")

    finite = np.isfinite(coords).all(axis=1)
    out = np.ones(len(coords), dtype=float)
    selected = np.full(len(coords), k_min, dtype=int)
    if finite.sum() < 2:
        return (out, selected) if return_k else out

    unique_coords, inverse, counts = np.unique(
        coords[finite], axis=0, return_inverse=True, return_counts=True
    )
    n_unique = len(unique_coords)
    if n_unique < 2:
        return (out, selected) if return_k else out

    max_effective = min(k_max, n_unique - 1)
    min_effective = min(k_min, max_effective)
    effective_ks = range(min_effective, max_effective + 1)
    tree = cKDTree(unique_coords)
    all_distances, _ = tree.query(unique_coords, k=max_effective + 1)
    all_distances = np.asarray(all_distances)

    best_lof = np.full(n_unique, -np.inf)
    best_k = np.full(n_unique, min_effective, dtype=int)
    for k_eff in effective_ks:
        k_distance = all_distances[:, k_eff]

        neighbourhoods = tree.query_ball_point(
            unique_coords, np.nextafter(k_distance, np.inf)
        )

        lrd = np.empty(n_unique, dtype=float)
        for u, neighbours in enumerate(neighbourhoods):
            neighbours = np.asarray(neighbours, dtype=int)
            multiplicity = counts[neighbours].astype(float)
            multiplicity[neighbours == u] -= 1.0
            keep = multiplicity > 0
            neighbours, multiplicity = neighbours[keep], multiplicity[keep]
            direct = np.linalg.norm(
                unique_coords[neighbours] - unique_coords[u], axis=1
            )
            reachability = np.maximum(k_distance[neighbours], direct)
            mean_reachability = np.average(reachability, weights=multiplicity)
            lrd[u] = 1.0 / mean_reachability

        current = np.empty(n_unique, dtype=float)
        for u, neighbours in enumerate(neighbourhoods):
            neighbours = np.asarray(neighbours, dtype=int)
            multiplicity = counts[neighbours].astype(float)
            multiplicity[neighbours == u] -= 1.0
            keep = multiplicity > 0
            neighbours, multiplicity = neighbours[keep], multiplicity[keep]
            current[u] = np.average(lrd[neighbours] / lrd[u], weights=multiplicity)

        improve = current > best_lof
        best_lof[improve] = current[improve]
        best_k[improve] = k_eff

    finite_rows = np.flatnonzero(finite)
    out[finite_rows] = best_lof[inverse]
    selected[finite_rows] = best_k[inverse]
    return (out, selected) if return_k else out


def score_law(x: np.ndarray, y: np.ndarray, k=DEFAULT_K_RANGE):
    C, theta = fit_power_law(x, y)
    yhat = C * np.power(np.maximum(x, 1e-12), theta)

    ol = out_of_line(y, yhat)

    close_to_line = np.isclose(y, yhat, rtol=1e-10, atol=1e-12)
    ol[close_to_line] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        coords = np.column_stack(
            [np.log(np.maximum(x, 1e-12)), np.log(np.maximum(y, 1e-12))]
        )
    k_min, k_max = _normalise_k_range(k)
    lf, lf_k = lof_k_distinct_range(coords, k_min, k_max, return_k=True)

    if not np.isfinite(ol).all() or not np.isfinite(lf).all():
        raise ValueError("OddBall score components must be finite")
    ol_n = ol / ol.max() if ol.max() > 0 else ol
    lf_n = lf / lf.max() if lf.max() > 0 else lf
    side = np.where(close_to_line, "on-line", np.where(y > yhat, "heavy", "light"))
    driver = np.where(
        close_to_line & (lf > 1.0 + 1e-12),
        "LOF (on-line)",
        np.where(lf_n > ol_n, "LOF-dominant", "line-dominant"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_residual = np.abs(np.log(np.maximum(y, 1e-12) / np.maximum(yhat, 1e-12)))
    return {
        "score": ol_n + lf_n,
        "out_line": ol,
        "lof": lf,
        "lof_k": lf_k,
        "line_component": ol_n,
        "lof_component": lf_n,
        "expected": yhat,
        "log_residual": log_residual,
        "score_driver": driver,
        "side": side,
        "C": C,
        "theta": theta,
        "k_range": (k_min, k_max),
    }


def score_features(feat: pd.DataFrame, k=DEFAULT_K_RANGE, laws=None) -> pd.DataFrame:
    out = feat.copy()
    if laws is None:
        laws = ("EDPL", "EWPL", "ELWPL") if "lambda_w" in feat else ("EDPL", "EWPL")
    laws = {law.upper() for law in laws}
    unknown = laws - {"EDPL", "EWPL", "ELWPL"}
    if unknown:
        raise ValueError(f"unknown laws: {sorted(unknown)}")
    if "ELWPL" in laws and "lambda_w" not in feat:
        raise ValueError("ELWPL requires a lambda_w feature")

    component_names = (
        "score",
        "side",
        "expected",
        "out_line",
        "lof",
        "lof_k",
        "line_component",
        "lof_component",
        "log_residual",
        "score_driver",
    )
    results = {}
    if "EDPL" in laws:
        results["edpl"] = score_law(
            feat["N"].to_numpy(float), feat["E"].to_numpy(float), k
        )
    if "EWPL" in laws:
        results["ewpl"] = score_law(
            feat["E"].to_numpy(float), feat["W"].to_numpy(float), k
        )
    if "ELWPL" in laws:
        results["elwpl"] = score_law(
            feat["W"].to_numpy(float), feat["lambda_w"].to_numpy(float), k
        )
    for prefix, result in results.items():
        for name in component_names:
            out[f"{prefix}_{name}"] = result[name]
        out.attrs[f"{prefix}_theta"] = result["theta"]
        out.attrs[f"{prefix}_C"] = result["C"]
    if results:
        out.attrs["lof_k_range"] = next(iter(results.values()))["k_range"]
    return out


def jaccard(a, b) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return float("nan")
    return len(A & B) / len(A | B)
