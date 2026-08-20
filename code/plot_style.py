"""Shared publication style and uncertainty helpers for paper figures."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# Colorblind-safe, print-friendly roles.  Meanings stay fixed across figures.
COLORS = {
    "core": "#00897B",
    "baseline": "#3F60D9",
    "risk": "#C63D2F",
    "orange": "#D97706",
    "ink": "#20243A",
    "grey": "#8A94A6",
    "light": "#D9DEE7",
    "grid": "#D7DBE2",
}


def apply_style(base: float = 9.0) -> None:
    """Apply a compact style that remains legible at ICLR column widths."""
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": base + 0.2,
            "axes.titleweight": "medium",
            "axes.titlepad": 7.0,
            "xtick.labelsize": base - 1.0,
            "ytick.labelsize": base - 1.0,
            "legend.fontsize": base - 1.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
            "errorbar.capsize": 2.4,
        }
    )


def grid(ax, axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.55, alpha=0.68)


def ci95_from_std(std, n: int):
    return 1.96 * np.asarray(std, dtype=float) / np.sqrt(float(n))


def mean_ci95(values, axis=-1):
    arr = np.asarray(values, dtype=float)
    mean = np.mean(arr, axis=axis)
    n = arr.shape[axis]
    half = 1.96 * np.std(arr, axis=axis, ddof=1) / np.sqrt(float(n))
    return mean, mean - half, mean + half


def bootstrap_mean_ci(values, reps: int = 10_000, seed: int = 0):
    """Percentile CI resampling complete independent units."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arr), size=(reps, len(arr)))
    means = arr[draws].mean(axis=1)
    return float(arr.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def errorbar_record(ax, y, record, color, marker="o", ms=4.6, zorder=3):
    estimate = float(record["estimate"])
    lower = float(record["lower"])
    upper = float(record["upper"])
    ax.errorbar(
        estimate,
        y,
        xerr=[[estimate - lower], [upper - estimate]],
        fmt=marker,
        ms=ms,
        color=color,
        ecolor=color,
        elinewidth=1.25,
        capsize=2.2,
        zorder=zorder,
    )
    return estimate


def annotate_x(ax, x, y, text=None, dx=4, color=None, fontsize=7.3, ha="left"):
    ax.annotate(
        f"{x:.3f}" if text is None else text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        ha=ha,
        fontsize=fontsize,
        color=COLORS["ink"] if color is None else color,
        clip_on=False,
    )
