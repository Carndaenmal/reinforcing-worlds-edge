#!/usr/bin/env python3
"""Plot the registered continual cue-MNIST supervised confirmation.

Run from anywhere:
    python3 code/plot_cue_mnist_confirmation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import COLORS, annotate_x, apply_style, errorbar_record, grid


HERE = Path(__file__).resolve().parent
SUMMARY = (
    HERE
    / "continual_cue_mnist_v4_confirmation_stats"
    / "confirmation_summary_supervised.json"
)
OUT = HERE.parent / "figs" / "fig_cue_mnist_confirmation.pdf"


def make_figure() -> Path:
    summary = json.loads(SUMMARY.read_text())
    prediction = summary["prediction"]
    mitigation = summary["mitigation"]

    apply_style(8.9)
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.35),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.08, 1.18], "wspace": 0.13},
    )

    contrasts = [
        (
            "Replay reward\n(0.950 vs 0.938)",
            mitigation["paired_certificate_advantage"],
        ),
        (
            "Prediction AP\n(0.384 vs 0.345)",
            prediction["paired_advantage"],
        ),
    ]
    ax[0].axvline(0, color=COLORS["risk"], linestyle="--", linewidth=1.0)
    for y, (label, record) in enumerate(contrasts):
        estimate = errorbar_record(
            ax[0], y, record, COLORS["core"], marker="D", ms=4.9
        )
        annotate_x(
            ax[0],
            estimate,
            y,
            text=rf"$+{estimate:.3f}$",
            fontsize=7.5,
        )
    ax[0].set_yticks(range(2), [row[0] for row in contrasts])
    ax[0].set_xlim(-0.006, 0.068)
    ax[0].set_xlabel(r"paired core advantage  $\uparrow$")
    ax[0].set_title("(a) Both registered endpoints pass", loc="left")
    grid(ax[0], "x")

    treatments = [
        ("Core-triggered replay", "certificate_reward", COLORS["core"]),
        ("Oracle", "oracle_reward", COLORS["ink"]),
        ("Accuracy-triggered replay", "accuracy_trigger_reward", COLORS["baseline"]),
        ("Random-matched replay", "random_matched_reward", COLORS["grey"]),
        ("Never intervene", "never_intervene_reward", COLORS["risk"]),
    ]
    oracle = mitigation["oracle_reward"]
    ax[1].axvline(
        oracle, color=COLORS["ink"], linestyle=":", linewidth=1.0, alpha=0.75
    )
    for y, (_, key, color) in enumerate(reversed(treatments)):
        value = float(mitigation[key])
        ax[1].scatter(value, y, color=color, marker="s", s=23, zorder=3)
        annotate_x(ax[1], value, y, text=f"{value:.3f}", fontsize=7.1)
    ax[1].set_yticks(range(len(treatments)), [row[0] for row in reversed(treatments)])
    ax[1].set_xlim(0.888, 0.957)
    ax[1].set_xlabel(r"whole-stream balanced reward  $\uparrow$")
    ax[1].set_title("(b) Core replay is near-oracle", loc="left")
    grid(ax[1], "x")

    fig.savefig(OUT)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(make_figure())
