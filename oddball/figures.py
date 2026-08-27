"""Project figures."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#004B98"
MIDBLUE = "#5B8DBF"
RED = "#C8102E"
ORANGE = "#E8820C"
GREY = "#9AA7B4"

WEIGHT_COLOURS = {
    "flow": BLUE,
    "packet": ORANGE,
    "byte": RED,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "axes.linewidth": 1.3,
        "axes.edgecolor": "#444444",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"    saved {name}.{{pdf,png}}")


def fig_k_curve(kc: pd.DataFrame, outdir="figures", name="fig_k_curve"):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    cols = {"flow-packet": BLUE, "flow-byte": MIDBLUE, "packet-byte": RED}
    for pair, s in kc.groupby("pair", sort=False):
        s = s.sort_values("K")
        ax.plot(
            s["K"],
            s["jaccard"],
            "o-",
            color=cols.get(pair, GREY),
            label=pair,
            lw=2,
            ms=5,
        )
    ax.set_xlabel("K (size of top-K anomaly set)")
    ax.set_ylabel("Jaccard overlap")
    ax.set_ylim(0, 0.30)
    ax.set_yticks(np.arange(0, 0.31, 0.05))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, outdir, name)


def fig_all_power_laws(
    scored_by_weight: dict, outdir="figures", name="fig_all_power_laws", topk=20
):
    specifications = {
        "EDPL": ("edpl", "N", "E", "egonet degree N", "egonet edges E", "E", "N"),
        "EWPL": ("ewpl", "E", "W", "egonet edges E", "egonet weight W", "W", "E"),
        "ELWPL": (
            "elwpl",
            "W",
            "lambda_w",
            "egonet weight W",
            r"weighted eigenvalue $\lambda_w$",
            r"\lambda_w",
            "W",
        ),
    }
    weights = [("n_flows", "Flow"), ("n_packets", "Packet"), ("n_bytes", "Byte")]
    fig = plt.figure(figsize=(16, 14))
    grid = fig.add_gridspec(3, 6)
    panels = [("EDPL", "n_flows", "Structure", fig.add_subplot(grid[0, 2:4]))]
    for row, law in enumerate(("EWPL", "ELWPL"), start=1):
        for column, (weight, label) in enumerate(weights):
            panels.append(
                (
                    law,
                    weight,
                    label,
                    fig.add_subplot(grid[row, 2 * column : 2 * column + 2]),
                )
            )

    for law, weight, weight_label, ax in panels:
        prefix, xcol, ycol, xlabel, ylabel, y_symbol, x_symbol = specifications[law]
        frame = scored_by_weight[weight]
        valid = frame[(frame[xcol] > 0) & (frame[ycol] > 0)]
        ax.scatter(
            valid[xcol],
            valid[ycol],
            s=6,
            color=BLUE,
            alpha=0.12,
            edgecolors="none",
            rasterized=True,
        )
        theta = frame.attrs[f"{prefix}_theta"]
        C = frame.attrs[f"{prefix}_C"]
        xs = np.geomspace(valid[xcol].min(), valid[xcol].max(), 160)
        ax.plot(
            xs,
            C * xs**theta,
            color="black",
            lw=1.8,
            label=rf"${y_symbol}={C:.3g}{x_symbol}^{{{theta:.3f}}}$",
        )
        top = frame.nlargest(topk, f"{prefix}_score")
        line = top[top[f"{prefix}_score_driver"] == "line-dominant"]
        lof = top[top[f"{prefix}_score_driver"].str.startswith("LOF")]
        ax.scatter(
            line[xcol],
            line[ycol],
            s=52,
            facecolors="none",
            edgecolors=RED,
            lw=1.5,
            label="line-driven",
        )
        ax.scatter(
            lof[xcol],
            lof[ycol],
            s=58,
            marker="D",
            facecolors="none",
            edgecolors=ORANGE,
            lw=1.5,
            label="LOF-driven",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{law} — {weight_label}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, outdir, name)


def fig_weight_correlation(edges, outdir="figures", name="fig_weight_correlation"):
    columns = ["n_flows", "n_packets", "n_bytes"]
    labels = ["Flow", "Packets", "Bytes"]
    correlation = edges[columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    for row in range(3):
        for column in range(3):
            value = correlation.iloc[row, column]
            colour = "white" if abs(value) >= 0.55 else "black"
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=14,
            )
    ax.set_title("Spearman correlation of raw edge weights")
    colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colourbar.set_label("Spearman correlation")
    fig.tight_layout()
    _save(fig, outdir, name)
    return correlation


def fig_composite_ewpl(
    scored, labels, outdir="figures", name="fig_composite_ewpl", topk=50
):
    methods = list(scored)
    ncols = 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), squeeze=False)
    for ax, method in zip(axes.ravel(), methods):
        frame = scored[method]
        valid = frame[(frame["E"] > 0) & (frame["W"] > 0)]
        ax.scatter(
            valid["E"],
            valid["W"],
            s=5,
            color=BLUE,
            alpha=0.12,
            edgecolors="none",
            rasterized=True,
        )
        theta = frame.attrs["ewpl_theta"]
        C = frame.attrs["ewpl_C"]
        xs = np.geomspace(valid["E"].min(), valid["E"].max(), 160)
        ax.plot(
            xs,
            C * xs**theta,
            color="black",
            lw=1.7,
            label=rf"$W={C:.3g}E^{{{theta:.3f}}}$",
        )
        top = frame.nlargest(topk, "ewpl_score")
        line = top[top["ewpl_score_driver"] == "line-dominant"]
        lof = top[top["ewpl_score_driver"].str.startswith("LOF")]
        ax.scatter(
            line["E"],
            line["W"],
            s=42,
            facecolors="none",
            edgecolors=RED,
            lw=1.2,
            label="line-driven",
        )
        ax.scatter(
            lof["E"],
            lof["W"],
            s=44,
            marker="D",
            facecolors="none",
            edgecolors=ORANGE,
            lw=1.2,
            label="LOF-driven",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(labels.get(method, method))
        ax.set_xlabel("egonet edges E")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes.ravel()[len(methods) :]:
        ax.set_axis_off()
    fig.supylabel("median-normalised composite egonet weight W", x=0.004)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    _save(fig, outdir, name)
