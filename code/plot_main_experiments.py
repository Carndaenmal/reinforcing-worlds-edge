import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import bootstrap_mean_ci
from make_oral_figures import (setup, title2, xgrid, lollirow, CORE, ORACLE, RETURN, MUTED, DANGER, INK, SUB, GRID, BANDT, HERE, OUT, CONTROL_SUMMARY)

def _ygrid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make_figure():
    """Combined Figure 2: E10 (top, 3 panels) over E11 (bottom, 2 panels) in ONE
    figure, so font scale, panel size, and spacing are consistent across rows."""
    import plot_lbf_core_erosion_exploratory as _LBF

    summary = json.loads(CONTROL_SUMMARY.read_text())
    per_seed = [
        json.loads(p.read_text())
        for p in sorted((CONTROL_SUMMARY.parent / "per_seed").glob("seed_*.json"))
    ]

    def record(section, key):
        it = summary[section][key]
        return it["estimate"], (it["lower"], it["upper"])

    expl = json.loads(
        (HERE / "lbf_v3_core_erosion_exploratory_stats"
         / "exploratory_core_erosion_stats.json").read_text()
    )
    pairs = expl["per_pairing"]
    baseline_values = np.array([r["return_slope_ap"] for r in pairs])
    erosion_values = np.array([r["erosion_3_ap"] for r in pairs])

    setup(9.5)
    fig = plt.figure(figsize=(7.9, 5.55))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.50, wspace=1.62)
    axT = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
           fig.add_subplot(gs[0, 4:6])]
    axB = [fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5])]

    # ---------- TOP ROW: E10 continual control ----------
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
        v, iv = record("average_precision", key)
        ap.append((label, v, color, hero, iv))
    lollirow(axT[0], ap, xmax=0.66, stems=True, hilite_label="Core risk")
    rnd = summary["average_precision"]["random"]["estimate"]
    axT[0].axvline(rnd, color=MUTED, ls=(0, (3, 3)), lw=1.0, zorder=1)
    axT[0].text(rnd, 6.62, "chance", color="#7A828E", fontsize=7.2, ha="center", va="bottom")
    axT[0].set_xlabel("average precision \u2191")
    axT[0].set_xticks([0.0, 0.2, 0.4, 0.6])
    xgrid(axT[0])
    title2(axT[0], "(a) Predicts failure", "far above every baseline")

    structural_seed = [
        sum(r["prediction"]["structural_lead_steps"]) / len(r["prediction"]["structural_lead_steps"])
        for r in per_seed
    ]
    return_seed = [
        sum(r["prediction"]["return_lead_steps"]) / len(r["prediction"]["return_lead_steps"])
        for r in per_seed
    ]
    r_ci = bootstrap_mean_ci(return_seed, seed=202610)
    s_ci = bootstrap_mean_ci(structural_seed, seed=202611)
    lead = [("Core risk", s_ci[0], CORE, s_ci[1:]), ("Return signal", r_ci[0], RETURN, r_ci[1:])]
    for y, (lab, val, col, (lo, hi)) in zip([1, 0], lead):
        axT[1].barh(y, val, height=0.6, color=col, zorder=2, edgecolor="white", linewidth=0.8)
        axT[1].errorbar(val, y, xerr=[[val - lo], [hi - val]], fmt="none", ecolor=INK,
                        elinewidth=1.15, capsize=2.3, zorder=3)
        inside = val >= 20
        axT[1].annotate(
            f"{val:.1f}",
            (val if inside else hi, y),
            xytext=(-7 if inside else 6, 0),
            textcoords="offset points",
            va="center",
            ha="right" if inside else "left",
            fontsize=8.6,
            fontweight="bold" if col == CORE else "normal",
            color="white" if inside else INK,
            zorder=4,
        )
    axT[1].set_yticks([1, 0], [r[0] for r in lead])
    axT[1].get_yticklabels()[1].set_color("#6C7480")
    axT[1].set_xlim(0, 104)
    axT[1].set_ylim(-0.6, 1.6)
    axT[1].text(40, 0.5, f"\u2248{s_ci[0] / r_ci[0]:.0f}\u00d7", ha="center", va="center",
                fontsize=9.0, color=CORE, fontweight="bold")
    axT[1].set_xlabel("steps before failure \u2191")
    axT[1].set_xticks([0, 40, 80])
    axT[1].tick_params(axis="y", pad=2)
    xgrid(axT[1])
    title2(axT[1], "(b) Warns earlier", "warns well ahead")

    reward_specs = [
        ("Core trigger", "certificate", CORE, True),
        ("Oracle", "oracle", ORACLE, False),
        ("Random-matched", "random_matched", MUTED, False),
        ("Never switch", "never", DANGER, False),
        ("Return trigger", "return_trigger", RETURN, False),
    ]
    rw = []
    for label, key, color, hero in reward_specs:
        v, iv = record("mean_reward", key)
        rw.append((label, v, color, hero, iv))
    lollirow(axT[2], rw, xmax=1.06, stems=False, hilite_label="Core trigger")
    axT[2].set_xlim(0.40, 1.06)
    axT[2].tick_params(axis="y", labelsize=7.2, pad=1)
    orc = summary["mean_reward"]["oracle"]["estimate"]
    axT[2].axvline(orc, color=ORACLE, ls=(0, (2, 2)), lw=1.0, alpha=0.8, zorder=1)
    axT[2].text(orc, 4.62, "oracle", color=ORACLE, fontsize=7.2, ha="center", va="bottom")
    axT[2].set_xlabel("mean reward \u2191")
    axT[2].set_xticks([0.5, 0.7, 0.9])
    xgrid(axT[2])
    title2(axT[2], "(c) Recovers reward", "\u2248 oracle; others collapse")

    # ---------- BOTTOM ROW: E11 LBF ----------
    for r in pairs:
        ys = [r["return_slope_ap"], r["erosion_3_ap"]]
        axB[0].plot([0, 1], ys, color="#D9DEE7", linewidth=1.0, zorder=1)
        axB[0].scatter([0, 1], ys, color=[RETURN, CORE], s=26, zorder=2)
    b_ci = bootstrap_mean_ci(baseline_values, seed=202612)
    e_ci = bootstrap_mean_ci(erosion_values, seed=202613)
    for x, ((est, lo, hi), color) in enumerate(zip([b_ci, e_ci], [RETURN, CORE])):
        axB[0].errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt="D", color=color,
                        ecolor=color, ms=6, elinewidth=1.5, capsize=2.6, zorder=4)
    axB[0].set_xticks([0, 1], ["Return\nslope", "Core\nerosion$_3$"])
    axB[0].set_xlim(-0.4, 1.4)
    axB[0].set_ylim(0.55, 1.045)
    axB[0].set_ylabel(r"average precision  $\uparrow$")
    ctr = expl["paired_contrast"]
    axB[0].text(
        0.5,
        0.558,
        rf"paired $\Delta=+{ctr['estimate']:.3f}$"
        + "\n"
        + rf"95\% CI $[{ctr['lower']:.3f},\,{ctr['upper']:.3f}]$",
        fontsize=7.8,
        linespacing=1.05,
        ha="center",
        va="bottom",
        fontweight="bold",
        color=CORE,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        zorder=5,
    )
    _ygrid(axB[0])
    title2(axB[0], "(d) Exploratory pairings", "erosion_3 beats return slope in 8/8")

    xs = list(range(1, 6))
    wm = expl["nearby_window_sensitivity_mean_ap"]
    ys = [wm[f"erosion_{w}"] for w in xs]
    lo_err, hi_err = [], []
    for w in xs:
        vals = erosion_values if w == 3 else _LBF._window_pair_values(w)
        _, lo, hi = bootstrap_mean_ci(vals, seed=202620 + w)
        mean = ys[w - 1]
        lo_err.append(mean - lo)
        hi_err.append(hi - mean)
    axB[1].errorbar(xs, ys, yerr=[lo_err, hi_err], color=CORE, marker="o", markersize=5,
                    linewidth=1.6, capsize=2.5, zorder=3)
    axB[1].scatter([3], [ys[2]], color=ORACLE, marker="D", s=34, zorder=4)
    base = expl["strongest_simple_baseline"]["mean_average_precision"]
    _, bl, bu = bootstrap_mean_ci(baseline_values, seed=202630)
    axB[1].axhspan(bl, bu, color=RETURN, alpha=0.10, lw=0)
    axB[1].axhline(base, color=RETURN, linestyle="--", linewidth=1.1)
    axB[1].text(5.45, base + 0.006, "return slope", color=RETURN, fontsize=7.6,
                va="bottom", ha="right")
    axB[1].set_xlim(0.75, 5.55)
    axB[1].set_ylim(0.60, 1.02)
    axB[1].set_xticks(xs)
    axB[1].set_xlabel("erosion window (checkpoints)")
    axB[1].set_ylabel(r"mean average precision  $\uparrow$")
    _ygrid(axB[1])
    title2(axB[1], "(e) Window sensitivity", "same eight development pairings")

    fig.subplots_adjust(left=0.078, right=0.985, top=0.935, bottom=0.075)
    fig.savefig(OUT / "fig_main_experiments.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print("wrote fig_main_experiments")
