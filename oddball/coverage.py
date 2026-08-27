"""Check the number of relative one-hour windows in each daily file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import NETFLOW_COLS, first_time


def coverage_day(path, day, chunksize=2_000_000):
    origin = first_time(path)
    observed = set()
    for chunk in pd.read_csv(
        path,
        names=NETFLOW_COLS,
        header=None,
        usecols=["Time"],
        chunksize=chunksize,
        compression="infer",
    ):
        observed.update(((chunk["Time"] - origin) // 3600).astype(int).unique())
    return {
        "day": day,
        "n_observed_hours": len(observed),
        "has_relative_hours_0_to_23": observed == set(range(24)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="..")
    parser.add_argument("--start-day", type=int, default=2)
    parser.add_argument("--end-day", type=int, default=33)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument(
        "--output",
        default="results/activity_by_relative_hour/daily_24_hour_coverage.csv",
    )
    args = parser.parse_args()

    rows = []
    for day in range(args.start_day, args.end_day + 1):
        path = Path(args.data_dir) / f"netflow_day-{day:02d}.bz2"
        row = coverage_day(path, day, args.chunksize)
        rows.append(row)
        print(
            f"Day {day:02d}: {row['n_observed_hours']} relative hours; "
            f"complete={row['has_relative_hours_0_to_23']}",
            flush=True,
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
