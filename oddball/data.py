"""NetFlow loading and edge aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

NETFLOW_COLS = [
    "Time",
    "Duration",
    "SrcDevice",
    "DstDevice",
    "Protocol",
    "SrcPort",
    "DstPort",
    "SrcPackets",
    "DstPackets",
    "SrcBytes",
    "DstBytes",
]

WEIGHT_COLS = ["n_flows", "n_packets", "n_bytes"]
WEIGHT_LABELS = {
    "n_flows": "flow",
    "n_packets": "packet",
    "n_bytes": "byte",
}


def first_time(path, has_header=False):
    df = pd.read_csv(
        path,
        names=None if has_header else NETFLOW_COLS,
        header=0 if has_header else None,
        nrows=1,
        compression="infer",
    )
    return int(df["Time"].iloc[0])


def load_window(
    path,
    start=None,
    end=None,
    has_header=False,
    nrows=None,
    relative=False,
    chunked=True,
    chunksize=2_000_000,
    assume_sorted=True,
):
    names = None if has_header else NETFLOW_COLS
    header = 0 if has_header else None

    if relative:
        t0 = first_time(path, has_header=has_header)
        start = None if start is None else t0 + start
        end = None if end is None else t0 + end

    read_kw = {"names": names, "header": header, "compression": "infer"}

    if (start is None and end is None) or nrows is not None or not chunked:
        df = pd.read_csv(path, nrows=nrows, **read_kw)
        if start is not None:
            df = df[df["Time"] >= start]
        if end is not None:
            df = df[df["Time"] < end]
        if df.empty:
            raise ValueError(f"No records in [{start}, {end})")
        return df.reset_index(drop=True)

    parts = []
    for ch in pd.read_csv(path, chunksize=chunksize, **read_kw):
        t = ch["Time"]
        sel = ch
        if start is not None:
            sel = sel[sel["Time"] >= start]
        if end is not None:
            sel = sel[sel["Time"] < end]
        if len(sel):
            parts.append(sel)
        if assume_sorted and end is not None and t.iloc[-1] >= end:
            break

    if not parts:
        raise ValueError(
            f"No records in [{start}, {end}). The file's first Time is "
            f"{first_time(path, has_header=has_header)} - LANL times are "
            f"offsets from the start of collection, so try relative=True."
        )
    return pd.concat(parts, ignore_index=True)


def build_edge_table(flows: pd.DataFrame) -> pd.DataFrame:
    f = flows

    if len(f) == 0:
        return pd.DataFrame(columns=["src", "dst"] + WEIGHT_COLS)

    a = f["SrcDevice"].astype(str).to_numpy()
    b = f["DstDevice"].astype(str).to_numpy()

    keep = a != b
    a, b = a[keep], b[keep]
    f = f[keep]

    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)

    packets = f["SrcPackets"].to_numpy(dtype=float) + f["DstPackets"].to_numpy(
        dtype=float
    )
    byts = f["SrcBytes"].to_numpy(dtype=float) + f["DstBytes"].to_numpy(dtype=float)

    tmp = pd.DataFrame(
        {
            "src": lo,
            "dst": hi,
            "n_flows": 1.0,
            "n_packets": packets,
            "n_bytes": byts,
        }
    )
    edges = tmp.groupby(["src", "dst"], as_index=False, sort=False).sum()

    for c in WEIGHT_COLS:
        v = edges[c].to_numpy(dtype=float)
        n_zero = int((v <= 0).sum())
        if n_zero:
            edges[c] = np.maximum(v, 1.0)
    return edges


def positive_edge_medians(edges: pd.DataFrame) -> dict[str, float]:
    medians = {}
    for column in WEIGHT_COLS:
        values = edges[column].to_numpy(dtype=float)
        positive = values[values > 0]
        medians[column] = float(np.median(positive)) if len(positive) else 1.0
    return medians
