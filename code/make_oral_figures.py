#!/usr/bin/env python3
"""
Oral-grade figure set for the Dec-MARL -> continual-RL paper.

Design goals: one idea per panel, a single hero (the core, in teal), muted
distractors, a plain-language takeaway above every panel, and one consistent
palette so colour == concept across the whole talk.

All plotted numbers are the exact values from the confirmation summaries
(continual_control_toy_v1 and continual_cue_mnist_v4).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from plot_style import bootstrap_mean_ci

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figs"
OUT.mkdir(exist_ok=True)
CONTROL_SUMMARY = (
    HERE
    / "continual_control_toy_v1_confirmation_stats"
    / "confirmation"
    / "confirmation_summary.json"
)
MNIST_SUMMARY = (
    HERE
    / "continual_cue_mnist_v4_confirmation_stats"
    / "confirmation_summary_supervised.json"
)

# ---- one palette, fixed roles -------------------------------------------------
CORE   = "#0E8A7E"   # ours / the core            (hero)
ORACLE = "#22283A"   # oracle / upper reference    (ink)
RETURN = "#3F63D6"   # return-based baseline       (main comparator)
MUTED  = "#B9C0CC"   # other baselines             (recede)
DANGER = "#D24A38"   # collapse / no-intervention
INK    = "#22283A"
SUB    = "#8A93A4"   # subtitle grey
GRID   = "#E6E9EF"
BANDT  = "#E4F3F0"   # faint teal highlight band
# route colours for the domain figure, deliberately OUTSIDE the data palette
TOP    = "#6D5FBE"   # top lane (purple)
BOT    = "#C77D24"   # bottom lane (amber)
BANDP  = "#EDEAF8"
BANDA  = "#F8EEDD"


def setup(base=9.5):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": base,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#5F6672",
        "xtick.labelsize": base - 1.0,
        "ytick.labelsize": base,
        "xtick.color": "#5F6672",
        "ytick.color": INK,
        "xtick.major.size": 3.0,
        "ytick.major.size": 0.0,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "svg.fonttype": "none",
    })


def title2(ax, big, small):
    """Two-tier panel title: bold phenomenon + grey detail, above the axes."""
    ax.text(0.0, 1.16, big, transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=INK, va="bottom", ha="left")
    ax.text(0.0, 1.05, small, transform=ax.transAxes, fontsize=8.0, color=SUB,
            va="bottom", ha="left")


def xgrid(ax):
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def lollirow(ax, rows, xmax, stem_from=0.0, stems=True, label_keys=None,
             fmt="{:.3f}", label_all=True, hilite_label=None):
    """Draw rows of ``(label, value, colour, is_hero[, (lower, upper)])``."""
    n = len(rows)
    ys = list(range(n))[::-1]  # first row at top
    for y, row in zip(ys, rows):
        lab, val, col, hero = row[:4]
        interval = row[4] if len(row) > 4 else None
        if hilite_label is not None and lab == hilite_label:
            ax.axhspan(y - 0.5, y + 0.5, color=BANDT, zorder=0)
        if stems:
            ax.hlines(y, stem_from, val, color=col, lw=(3.0 if hero else 1.6),
                      alpha=(1.0 if hero else 0.85), zorder=2)
        ax.scatter(val, y, s=(120 if hero else 55), color=col,
                   edgecolor="white", linewidth=0.9, zorder=3)
        if interval is not None:
            lo, hi = interval
            ax.errorbar(
                val,
                y,
                xerr=[[val - lo], [hi - val]],
                fmt="none",
                ecolor=col,
                elinewidth=1.35,
                capsize=2.3,
                zorder=2.8,
            )
        show = label_all or (label_keys and lab in label_keys)
        if show:
            ax.annotate(fmt.format(val), (val, y), xytext=(7, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8.4, fontweight=("bold" if hero else "normal"),
                        color=(col if hero else INK))
    ax.set_yticks(ys, [r[0] for r in rows])
    for y, r in zip(ys, rows):
        if not r[3]:
            ax.get_yticklabels()[ys.index(y)].set_color("#6C7480")
    ax.set_xlim(0 if stem_from == 0 else stem_from, xmax)
    ax.set_ylim(-0.6, n - 0.4)


# ============================================================ FIG 1: continual control
def fig_continual_control():
    summary = json.loads(CONTROL_SUMMARY.read_text())
    per_seed = [
        json.loads(path.read_text())
        for path in sorted((CONTROL_SUMMARY.parent / "per_seed").glob("seed_*.json"))
    ]

    def record(section, key):
        item = summary[section][key]
        return item["estimate"], (item["lower"], item["upper"])

    setup()
    fig, ax = plt.subplots(1, 3, figsize=(7.9, 2.85),
                           gridspec_kw={"width_ratios": [1.32, 0.9, 1.22], "wspace": 0.55})

    # (a) failure prediction --------------------------------------------------
    ap_specs = [
        ("Core risk", "structural_risk", CORE, True),
        ("Return slope", "return_slope", RETURN, False),
        ("State magnitude", "state_magnitude", MUTED, False),
        ("Return level", "return_level", MUTED, False),
        ("Random", "random", MUTED, False),
        ("Checkpoint time", "checkpoint_time", MUTED, False),
        ("Transition CUSUM", "transition_cusum", MUTED, False),
    ]
    ap = []
    for label, key, color, hero in ap_specs:
        value, interval = record("average_precision", key)
        ap.append((label, value, color, hero, interval))
    lollirow(ax[0], ap, xmax=0.66, stems=True, hilite_label="Core risk")
    random_ap = summary["average_precision"]["random"]["estimate"]
    ax[0].axvline(random_ap, color=MUTED, ls=(0, (3, 3)), lw=1.0, zorder=1)
    ax[0].text(random_ap, 6.62, "chance", color="#7A828E", fontsize=7.2, ha="center", va="bottom")
    ax[0].set_xlabel("average precision ↑")
    ax[0].set_xticks([0.0, 0.2, 0.4, 0.6])
    xgrid(ax[0])
    title2(ax[0], "(a) Predicts failure", "far above every baseline")

    # (b) warning lead --------------------------------------------------------
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
    return_ci = bootstrap_mean_ci(return_seed, seed=202610)
    structural_ci = bootstrap_mean_ci(structural_seed, seed=202611)
    lead = [
        ("Core risk", structural_ci[0], CORE, structural_ci[1:]),
        ("Return signal", return_ci[0], RETURN, return_ci[1:]),
    ]
    ys = [1, 0]
    for y, (lab, val, col, (lo, hi)) in zip(ys, lead):
        ax[1].barh(y, val, height=0.6, color=col, zorder=2,
                   edgecolor="white", linewidth=0.8)
        ax[1].errorbar(
            val,
            y,
            xerr=[[val - lo], [hi - val]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.15,
            capsize=2.3,
            zorder=3,
        )
        ax[1].annotate(f"{val:.1f}", (val, y), xytext=(6, 0), textcoords="offset points",
                       va="center", ha="left", fontsize=8.8,
                       fontweight=("bold" if col == CORE else "normal"),
                       color=(CORE if col == CORE else INK))
    ax[1].set_yticks(ys, [r[0] for r in lead])
    ax[1].get_yticklabels()[1].set_color("#6C7480")
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(-0.6, 1.6)
    lead_ratio = structural_ci[0] / return_ci[0]
    ax[1].text(40, 0.5, f"≈{lead_ratio:.0f}×", ha="center", va="center",
               fontsize=9.0, color=CORE, fontweight="bold")
    ax[1].set_xlabel("steps before failure ↑")
    ax[1].set_xticks([0, 40, 80])
    xgrid(ax[1])
    title2(ax[1], "(b) Warns earlier", "warns well ahead")

    # (c) control reward ------------------------------------------------------
    reward_specs = [
        ("Core trigger", "certificate", CORE, True),
        ("Oracle", "oracle", ORACLE, False),
        ("Random-matched", "random_matched", MUTED, False),
        ("Never switch", "never", DANGER, False),
        ("Return trigger", "return_trigger", RETURN, False),
    ]
    rw = []
    for label, key, color, hero in reward_specs:
        value, interval = record("mean_reward", key)
        rw.append((label, value, color, hero, interval))
    lollirow(ax[2], rw, xmax=1.06, stems=False, hilite_label="Core trigger")
    ax[2].set_xlim(0.40, 1.06)
    oracle = summary["mean_reward"]["oracle"]["estimate"]
    ax[2].axvline(oracle, color=ORACLE, ls=(0, (2, 2)), lw=1.0, alpha=0.8, zorder=1)
    ax[2].text(oracle, 4.62, "oracle", color=ORACLE, fontsize=7.2, ha="center", va="bottom")
    ax[2].set_xlabel("mean reward ↑")
    ax[2].set_xticks([0.5, 0.7, 0.9])
    xgrid(ax[2])
    title2(ax[2], "(c) Recovers reward", "≈ oracle; others collapse")

    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.11, right=0.975, wspace=0.62)
    fig.savefig(OUT / "fig_continual_control.pdf")
    plt.close(fig)


# ============================================================ FIG 2: cue-MNIST
def fig_cue_mnist():
    summary = json.loads(MNIST_SUMMARY.read_text())
    prediction = summary["prediction"]
    mitigation = summary["mitigation"]
    setup()
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.75),
                           gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.5})

    # (a) both registered endpoints clear zero -------------------------------
    adv = [
        (
            "Prediction AP",
            prediction["paired_advantage"]["estimate"],
            (
                prediction["paired_advantage"]["lower"],
                prediction["paired_advantage"]["upper"],
            ),
        ),
        (
            "Replay reward",
            mitigation["paired_certificate_advantage"]["estimate"],
            (
                mitigation["paired_certificate_advantage"]["lower"],
                mitigation["paired_certificate_advantage"]["upper"],
            ),
        ),
    ]
    ys = [1, 0]
    ax[0].axvspan(0, 0.078, color="#EAF4E7", zorder=0)         # "pass" zone
    ax[0].axvline(0, color=DANGER, ls=(0, (4, 3)), lw=1.1, zorder=1)
    for y, (lab, val, (lo, hi)) in zip(ys, adv):
        ax[0].errorbar(val, y, xerr=[[val - lo], [hi - val]], fmt="D", ms=7,
                       color=CORE, ecolor=CORE, elinewidth=1.6, capsize=2.6, zorder=3,
                       markeredgecolor="white", markeredgewidth=0.8)
        ax[0].annotate(f"+{val:.3f}", (hi, y), xytext=(8, 0), textcoords="offset points",
                       va="center", ha="left", fontsize=8.6, fontweight="bold", color=CORE)
    ax[0].set_yticks(ys, [r[0] for r in adv])
    ax[0].set_xlim(-0.008, 0.078)
    ax[0].set_ylim(-0.6, 1.6)
    ax[0].set_xticks([0.0, 0.02, 0.04])
    ax[0].set_xlabel("paired core advantage   (↑ better)")
    xgrid(ax[0])
    title2(ax[0], "(a) Both endpoints pass", "frozen confirmation endpoints clear zero")

    # (b) near-oracle reward --------------------------------------------------
    rw = [
        ("Core-triggered replay", mitigation["certificate_reward"], CORE, True),
        ("Oracle", mitigation["oracle_reward"], ORACLE, False),
        ("Accuracy-triggered", mitigation["accuracy_trigger_reward"], RETURN, False),
        ("Random-matched", mitigation["random_matched_reward"], MUTED, False),
        ("Never intervene", mitigation["never_intervene_reward"], DANGER, False),
    ]
    lollirow(ax[1], rw, xmax=0.957, stems=False, hilite_label="Core-triggered replay")
    ax[1].set_xlim(0.888, 0.957)
    oracle = mitigation["oracle_reward"]
    ax[1].axvline(oracle, color=ORACLE, ls=(0, (2, 2)), lw=1.0, alpha=0.8, zorder=1)
    ax[1].text(oracle, 4.62, "oracle", color=ORACLE, fontsize=7.2, ha="center", va="bottom")
    ax[1].set_xticks([0.90, 0.92, 0.94])
    ax[1].set_xlabel("whole-stream balanced reward   (↑ better)")
    xgrid(ax[1])
    title2(ax[1], "(b) Near-oracle reward", "core replay lands on the oracle")

    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.16, right=0.985)
    fig.savefig(OUT / "fig_cue_mnist_confirmation.pdf")
    plt.close(fig)


# ============================================================ FIG 3: SRC schematic
def fig_src_schematic():
    setup()
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.axis("off")
    ax.set_xlim(-0.8, 9.8)
    ax.set_ylim(-2.3, 2.5)

    # faint lane bands
    ax.add_patch(FancyBboxPatch((1.4, 0.55), 5.2, 0.9, boxstyle="round,pad=0.15,rounding_size=0.4",
                                fc=BANDP, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((1.4, -1.45), 5.2, 0.9, boxstyle="round,pad=0.15,rounding_size=0.4",
                                fc=BANDA, ec="none", zorder=0))

    def node(x, y, label, color, r=0.34, core=False):
        if core:
            ax.add_patch(Circle((x, y), r + 0.11, fc="none", ec=CORE, lw=2.4, zorder=3))
        ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=2.2, zorder=4))
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5, color=INK, zorder=5)

    def arrow(x1, y1, x2, y2, color, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     color=color, lw=2.0, connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=15, shrinkB=15, zorder=2))

    tx = [2, 4, 6]
    # arrows top
    arrow(0, 0, tx[0], 1.0, TOP, rad=0.22)
    for i in range(2):
        arrow(tx[i], 1.0, tx[i + 1], 1.0, TOP)
    arrow(tx[-1], 1.0, 8, 0, TOP, rad=0.22)
    # arrows bottom
    arrow(0, 0, tx[0], -1.0, BOT, rad=-0.22)
    for i in range(2):
        arrow(tx[i], -1.0, tx[i + 1], -1.0, BOT)
    arrow(tx[-1], -1.0, 8, 0, BOT, rad=-0.22)

    for i, x in enumerate(tx):
        node(x, 1.0, rf"$t_{i+1}$", TOP)
        node(x, -1.0, rf"$b_{i+1}$", BOT)
    node(0, 0, r"$s_0$", CORE, core=True)
    node(8, 0, r"$g$", CORE, core=True)

    ax.text(0.6, 1.95, "top lane — peer holds switch, rock intact", color=TOP, fontsize=8.6, ha="left")
    ax.text(0.6, -2.02, "bottom lane — peer clears rock", color=BOT, fontsize=8.6, ha="left")
    ax.text(4, 0.0, "clearing the rock closes the top lane", ha="center", va="center",
            fontsize=8.4, color=DANGER,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=DANGER, lw=1.1))

    # the payoff annotation: shared endpoints ARE the invariant core
    ax.annotate("shared by every success\n= invariant core  {$s_0$, $g$}",
                xy=(8, 0), xytext=(8.7, 1.5), fontsize=8.8, color=CORE, ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=CORE, lw=1.2,
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(0, 0), xytext=(1.0, 1.55),
                arrowprops=dict(arrowstyle="-", color=CORE, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))

    ax.text(4, 2.3, "Switch–Rock Corridor  (length $L=3$)", ha="center", fontsize=11,
            fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUT / "fig_env_schematic.pdf")
    plt.close(fig)


# ============================================================ FIG 4: reduction concept
def fig_reduction():
    setup()
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)

    def rbox(x, y, w, h, fc, ec, lw=1.6, ls="-", z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.2,rounding_size=1.4",
                     fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def arrow(x1, y1, x2, y2, color=INK, lw=1.8, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=1, shrinkB=1, zorder=3))

    # --- left inset: one Dec-MARL episode
    ax.text(11.5, 39.5, "Dec-MARL episode", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    rbox(2, 20, 19, 17, "none", "#8A93A4", lw=1.1, ls=(0, (3, 3)), z=1)
    rbox(4.5, 27, 8, 6.5, "#E4F3F0", CORE, lw=1.6)
    ax.text(8.5, 30.2, "focal", ha="center", va="center", fontsize=9.5, color="#0B5C54")
    for yy in (30.5, 23.0):
        rbox(13.5, yy, 6.5, 4.6, "#EEF0F4", "#9AA2AE", lw=1.2)
        ax.text(16.75, yy + 2.3, "peer", ha="center", va="center", fontsize=8.4, color="#5A626E")

    arrow(21.5, 28.5, 27.5, 28.5, INK, lw=1.8)
    ax.text(48, 41.0, "peers learn  (η)", ha="center", fontsize=9.2, color=SUB)

    # --- filmstrip of induced MDPs
    cards = [(29, "$M_1$", "episode 1"), (52, "$M_2$", "episode 2"), (75, "$M_3$", "episode 3")]
    cw, ch, cy = 19, 15, 21
    for cx, mlab, elab in cards:
        rbox(cx, cy, cw, ch, "white", "#9AA2AE", lw=1.2, z=2)
        ax.text(cx + 2.2, cy + ch - 3.2, mlab, ha="left", va="center", fontsize=10.5, color=INK)
        ax.text(cx + 2.2, cy + ch - 6.6, elab, ha="left", va="center", fontsize=8.0, color=SUB)
        s = (cx + 3.5, cy + 4.3); m = (cx + cw / 2, cy + 4.3); g = (cx + cw - 3.5, cy + 4.3)
        arrow(s[0] + 0.9, s[1], m[0] - 0.9, m[1], "#8A93A4", lw=1.3)
        arrow(m[0] + 0.9, m[1], g[0] - 0.9, g[1], "#8A93A4", lw=1.3)
        ax.add_patch(Circle(s, 0.75, fc="#8A93A4", ec="none", zorder=4))
        ax.add_patch(Circle(m, 0.75, fc="#8A93A4", ec="none", zorder=4))
        ax.add_patch(Circle(g, 0.95, fc=CORE, ec="none", zorder=4))
    # drift arrows between cards
    arrow(48.2, cy + ch / 2, 51.6, cy + ch / 2, "#8A93A4", lw=1.4)
    arrow(71.2, cy + ch / 2, 74.6, cy + ch / 2, "#8A93A4", lw=1.4)
    # dashed leaders down to the core rail
    for cx in (29 + cw / 2, 52 + cw / 2, 75 + cw / 2):
        ax.plot([cx, cx], [cy - 0.5, 15.6], color="#B9C0CC", lw=0.9, ls=(0, (2, 2)), zorder=1)

    # --- certified positive-margin core rail
    rbox(29, 10.5, 46, 5.2, "#CDECE6", CORE, lw=1.4, z=2)
    ax.text(31, 13.1, "positive-margin core candidate", ha="left", va="center", fontsize=9.1,
            color="#0B5C54", fontweight="bold")
    rbox(76, 10.5, 18, 5.2, "#E4F3F0", CORE, lw=1.0, ls=(0, (2, 2)), z=1)
    ax.text(52, 6.3, "coverage margin persists under bounded peer drift", ha="center",
            fontsize=8.6, color=SUB)
    ax.text(85, 6.3, r"certified survival $\Omega(1/\eta)$", ha="center", fontsize=8.6,
            color=DANGER, fontweight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "fig_reduction.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_continual_control()
    fig_cue_mnist()
    fig_src_schematic()
    fig_reduction()
    print("wrote figures to", OUT)
