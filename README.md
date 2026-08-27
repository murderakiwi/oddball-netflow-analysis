# OddBall NetFlow analysis

Weighted graph anomaly detection using hourly LANL NetFlow graphs.

## Contents

- [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb): complete analysis
- [`oddball/`](oddball/): data processing, scoring and experiment code

## Setup

```bash
python -m pip install -r requirements.txt
```

Place `netflow_day-02.bz2` through `netflow_day-33.bz2` in the project root.
The data files are not included in this repository.

Run the notebook from the project root. It performs the descriptive analysis,
30-day fit comparison and Day-33 controlled replacement experiment.

The replacement experiment can also be run directly:

```bash
python -m oddball.replacement \
  --data-dir . \
  --history-start-day 3 --history-end-day 32 --test-day 33 \
  --start 0 --end 3600 \
  --history-top-k 100 --top-k 50 \
  --trials 100 --seed 2026 \
  --lof-k-min 10 --lof-k-max 50 \
  --outdir results/replacement/hour00
```
