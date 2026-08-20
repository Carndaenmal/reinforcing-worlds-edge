#!/usr/bin/env python3
"""Plot the LBF post-hoc core-erosion analysis.

The eight peer-learning trajectories are excluded development data.  The
three-checkpoint erosion statistic was selected on these trajectories, so the
resulting panels are exploratory and must not be described as confirmation.

Run from anywhere:
    python3 code/plot_lbf_core_erosion_exploratory.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import COLORS, apply_style, bootstrap_mean_ci, grid


HERE = Path(__file__).resolve().parent
DATA = HERE / "lbf_v3_core_erosion_exploratory_stats"
OUT = HERE.parent / "figs" / "fig_lbf_core_erosion_exploratory.pdf"


def _average_precision(labels, scores):
    """Average precision with threshold ties handled as one operating point."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positive = labels.sum()
    if positive <= 0:
        return np.nan
    previous_recall = 0.0
    value = 0.0
    for threshold in np.unique(scores)[::-1]:
        selected = scores >= threshold
        true_positive = labels[selected].sum()
        recall = true_positive / positive
        precision = true_positive / selected.sum()
        value += (recall - previous_recall) * precision
        previous_recall = recall
    return float(value)


def _window_pair_values(window):
    values = []
    for pairing in range(5000, 5008):
        record = json.loads((DATA / f"pairing_{pairing}" / "summary.json").read_text())
        coverage = np.asarray([row["coverage_lcb_uL3"] for row in record["rows"]])
        labels = np.asarray(record["prediction_labels"], dtype=float)
        eligible = np.flatnonzero(np.isfinite(labels))
        scores = [
            max(0.0, coverage[t - window] - coverage[t]) if t >= window else 0.0
            for t in eligible
        ]
        values.append(_average_precision(labels[eligible], scores))
    return np.asarray(values)


def make_figure() -> Path:
    exploratory = json.loads((DATA / "exploratory_core_erosion_stats.json").read_text())
    pairs = exploratory["per_pairing"]
    windows = exploratory["nearby_window_sensitivity_mean_ap"]

    apply_style(9.5)
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.65),
                           gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.34})

    def _t2(a, big, small):
        a.text(0.0, 1.15, big, transform=a.transAxes, fontsize=11, fontweight="bold",
               color=COLORS["ink"], va="bottom", ha="left")
        a.text(0.0, 1.04, small, transform=a.transAxes, fontsize=8.0, color="#8A93A4",
               va="bottom", ha="left")

    # Post-hoc paired comparison.  Lines emphasize that trajectories, not
    # checkpoints, are the independent units.
    for row in pairs:
        ys = [row["return_slope_ap"], row["erosion_3_ap"]]
        ax[0].plot([0, 1], ys, color=COLORS["light"], linewidth=1.0, zorder=1)
        ax[0].scatter(
            [0, 1], ys, color=[COLORS["baseline"], COLORS["core"]], s=17, zorder=2
        )
    means = [
        exploratory["strongest_simple_baseline"]["mean_average_precision"],
        exploratory["primary_exploratory_score"]["mean_average_precision"],
    ]
    baseline_values = np.asarray([row["return_slope_ap"] for row in pairs])
    erosion_values = np.asarray([row["erosion_3_ap"] for row in pairs])
    endpoint_cis = [
        bootstrap_mean_ci(baseline_values, reps=10_000, seed=202612),
        bootstrap_mean_ci(erosion_values, reps=10_000, seed=202613),
    ]
    for x, ((estimate, lower, upper), color) in enumerate(
        zip(endpoint_cis, [COLORS["baseline"], COLORS["core"]])
    ):
        ax[0].errorbar(
            x,
            estimate,
            yerr=[[estimate - lower], [upper - estimate]],
            fmt="D",
            color=color,
            ecolor=color,
            ms=5.0,
            elinewidth=1.4,
            capsize=2.5,
            zorder=4,
        )
    ax[0].set_xticks([0, 1], ["Return\nslope", "Core\nerosion$_3$"])
    ax[0].set_ylim(0.58, 1.045)
    ax[0].set_ylabel("average precision")
    _t2(ax[0], "(a) Exploratory pairings", "erosion_3 beats return slope in 8/8")
    grid(ax[0], "y")
    contrast = exploratory["paired_contrast"]
    ax[0].text(
        0.5,
        0.600,
        rf"paired $\Delta=+{contrast['estimate']:.3f}$  "
        rf"95% CI $[{contrast['lower']:.3f},\,{contrast['upper']:.3f}]$",
        fontsize=8.1,
        ha="center",
        va="bottom",
        fontweight="bold",
        color=COLORS["core"],
    )

    # Nearby-window sensitivity is descriptive because every window uses the
    # same development trajectories as the selected erosion_3 score.
    xs = list(range(1, 6))
    ys = [windows[f"erosion_{window}"] for window in xs]
    window_cis = []
    for window in xs:
        values = erosion_values if window == 3 else _window_pair_values(window)
        _, lower, upper = bootstrap_mean_ci(values, reps=10_000, seed=202620 + window)
        window_cis.append((lower, upper))
    lower_err = [mean - bounds[0] for mean, bounds in zip(ys, window_cis)]
    upper_err = [bounds[1] - mean for mean, bounds in zip(ys, window_cis)]
    ax[1].errorbar(
        xs,
        ys,
        yerr=[lower_err, upper_err],
        color=COLORS["core"],
        marker="o",
        markersize=4.2,
        linewidth=1.45,
        capsize=2.3,
        label="core erosion (pair-bootstrap 95% CI)",
    )
    ax[1].scatter([3], [ys[2]], color=COLORS["ink"], marker="D", s=28, zorder=4)
    baseline = exploratory["strongest_simple_baseline"]["mean_average_precision"]
    _, baseline_lower, baseline_upper = bootstrap_mean_ci(
        baseline_values, reps=10_000, seed=202630
    )
    ax[1].axhspan(
        baseline_lower, baseline_upper, color=COLORS["baseline"], alpha=0.10, lw=0
    )
    ax[1].axhline(baseline, color=COLORS["baseline"], linestyle="--", linewidth=1.1)
    ax[1].text(
        5.48,
        baseline + 0.006,
        "return slope",
        color=COLORS["baseline"],
        fontsize=7.6,
        va="bottom",
        ha="right",
    )
    ax[1].set_xlim(0.75, 5.55)
    ax[1].set_ylim(0.60, 1.02)
    ax[1].set_xticks(xs)
    ax[1].set_xlabel("erosion window (checkpoints)")
    ax[1].set_ylabel("mean average precision")
    _t2(ax[1], "(b) Window sensitivity", "same eight development pairings")
    grid(ax[1], "y")

    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.095, right=0.975)
    fig.savefig(OUT)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(make_figure())
