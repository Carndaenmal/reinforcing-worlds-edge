#!/usr/bin/env python3
"""Plot the registered single-agent continual-control confirmation statistics.

The source archive contains only compact per-stream statistics, not per-step
arrays.  This script therefore reads the frozen aggregate estimates and their
whole-stream bootstrap intervals directly from ``confirmation_summary.json``.

Run from anywhere:
    python3 code/plot_continual_control_confirmation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import (
    COLORS,
    annotate_x,
    apply_style,
    bootstrap_mean_ci,
    errorbar_record,
    grid,
)


HERE = Path(__file__).resolve().parent
SUMMARY = (
    HERE
    / "continual_control_toy_v1_confirmation_stats"
    / "confirmation"
    / "confirmation_summary.json"
)
OUT = HERE.parent / "figs" / "fig_continual_control.pdf"


def make_figure() -> Path:
    summary = json.loads(SUMMARY.read_text())
    seed_dir = SUMMARY.parent / "per_seed"
    per_seed = [json.loads(path.read_text()) for path in sorted(seed_dir.glob("seed_*.json"))]

    apply_style(8.8)
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(7.5, 2.35),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.25, 0.88, 1.18], "wspace": 0.16},
    )

    predictors = [
        ("Core risk", "structural_risk", COLORS["core"]),
        ("Return slope", "return_slope", COLORS["baseline"]),
        ("State magnitude", "state_magnitude", COLORS["grey"]),
        ("Return level", "return_level", COLORS["grey"]),
        ("Random", "random", COLORS["grey"]),
        ("Checkpoint time", "checkpoint_time", COLORS["grey"]),
        ("Transition CUSUM", "transition_cusum", COLORS["grey"]),
    ]
    for y, (_, key, color) in enumerate(reversed(predictors)):
        estimate = errorbar_record(ax[0], y, summary["average_precision"][key], color)
        if key in {"structural_risk", "return_slope"}:
            annotate_x(ax[0], estimate, y, text=f"{estimate:.3f}", fontsize=7.0)
    ax[0].set_yticks(range(len(predictors)), [p[0] for p in reversed(predictors)])
    ax[0].set_xlim(0.0, 0.68)
    ax[0].set_xlabel(r"average precision  $\uparrow$")
    ax[0].set_title("(a) Failure prediction", loc="left")
    grid(ax[0], "x")

    structural_seed = [
        sum(row["prediction"]["structural_lead_steps"])
        / len(row["prediction"]["structural_lead_steps"])
        for row in per_seed
    ]
    return_seed = [
        sum(row["prediction"]["return_lead_steps"])
        / len(row["prediction"]["return_lead_steps"])
        for row in per_seed
    ]
    lead_records = [
        bootstrap_mean_ci(return_seed, seed=202610),
        bootstrap_mean_ci(structural_seed, seed=202611),
    ]
    lead_names = ["Return signal", "Core risk"]
    for y, ((estimate, lower, upper), color) in enumerate(
        zip(lead_records, [COLORS["baseline"], COLORS["core"]])
    ):
        record = {"estimate": estimate, "lower": lower, "upper": upper}
        errorbar_record(ax[1], y, record, color, marker="D", ms=4.8)
        annotate_x(ax[1], estimate, y, text=f"{estimate:.1f}", fontsize=7.1)
    ax[1].set_yticks(range(2), lead_names)
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel(r"steps before failure  $\uparrow$")
    ax[1].set_title("(b) Warning lead", loc="left")
    grid(ax[1], "x")

    treatments = [
        ("Core trigger", "certificate", COLORS["core"]),
        ("Oracle", "oracle", COLORS["ink"]),
        ("Random-matched", "random_matched", COLORS["grey"]),
        ("Never switch", "never", COLORS["risk"]),
        ("Return trigger", "return_trigger", COLORS["baseline"]),
    ]
    for y, (_, key, color) in enumerate(reversed(treatments)):
        estimate = errorbar_record(
            ax[2], y, summary["mean_reward"][key], color, marker="s", ms=4.4
        )
        annotate_x(ax[2], estimate, y, text=f"{estimate:.3f}", fontsize=6.8)
    ax[2].set_yticks(range(len(treatments)), [p[0] for p in reversed(treatments)])
    ax[2].set_xlim(0.40, 1.08)
    ax[2].set_xlabel(r"whole-stream mean reward  $\uparrow$")
    ax[2].set_title("(c) Control reward", loc="left")
    grid(ax[2], "x")

    fig.savefig(OUT)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(make_figure())
