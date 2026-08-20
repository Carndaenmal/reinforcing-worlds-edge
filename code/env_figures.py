"""Environment figures: (1) schematic of SRC state/transition structure,
(2) a small rendered gridworld instance showing the two routes and the switch/rock."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

OUT = "../figs/"
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight"})
INK = "#22223b"; TOP = "#6D5FBE"; BOT = "#C77D24"; GOAL = "#0E8A7E"; ROCK = "#8d5a2b"; SW = "#c1121f"


# ================================================= (1) schematic
def schematic():
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.axis("off"); ax.set_xlim(-0.5, 9.5); ax.set_ylim(-1.6, 1.8)

    def node(x, y, label, color, r=0.30):
        ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=2, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=INK, zorder=4)

    def arrow(x1, y1, x2, y2, color, style="-|>", rad=0.0, lw=1.6, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=11, color=color, lw=lw, ls=ls,
                     connectionstyle=f"arc3,rad={rad}", zorder=2,
                     shrinkA=13, shrinkB=13))

    node(0, 0, r"$s_0$", INK)
    tx = [2, 4, 6]
    for i, x in enumerate(tx):
        node(x, 1.0, rf"$t_{i+1}$", TOP)
    for i, x in enumerate(tx):
        node(x, -1.0, rf"$b_{i+1}$", BOT)
    node(8, 0, r"$g$", GOAL)

    arrow(0, 0, tx[0], 1.0, TOP, rad=0.25)
    for i in range(len(tx) - 1):
        arrow(tx[i], 1.0, tx[i + 1], 1.0, TOP)
    arrow(tx[-1], 1.0, 8, 0, TOP, rad=0.25)
    arrow(0, 0, tx[0], -1.0, BOT, rad=-0.25)
    for i in range(len(tx) - 1):
        arrow(tx[i], -1.0, tx[i + 1], -1.0, BOT)
    arrow(tx[-1], -1.0, 8, 0, BOT, rad=-0.25)

    ax.text(1.0, 1.55, "top gate: peer holds switch,  rock intact",
            ha="center", fontsize=6.6, color=TOP)
    ax.text(1.0, -1.62, "bottom gate: peer clears rock",
            ha="center", fontsize=6.6, color=BOT)
    ax.text(4, 0.02, "clearing rock disables top gate", ha="center",
            fontsize=6.6, color=SW,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=SW, lw=0.8))
    ax.text(4, 1.75, "Switch-Rock Corridor   (length L = 3)", ha="center",
            fontsize=9, color=INK)
    fig.savefig(OUT + "fig_env_schematic.pdf"); plt.close(fig)


# ================================================= (2) rendered gridworld instance
def gridworld():
    L = 3
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.set_xlim(-0.5, L + 2.5); ax.set_ylim(-0.5, 2.5); ax.set_aspect("equal"); ax.axis("off")

    def cell(x, y, fc="white", ec="#cccccc"):
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fc, edgecolor=ec, lw=1.2, zorder=1))

    for x in range(L + 2):
        cell(x, 1, "#f4f4f8")
    for x in range(1, L + 1):
        cell(x, 2, "#EDEAF8")
        cell(x, 0, "#F8EEDD")

    ax.text(0.5, 1.5, "S", ha="center", va="center", fontsize=13, color=INK, weight="bold")
    ax.add_patch(Rectangle((L + 1, 1), 1, 1, facecolor=GOAL, alpha=0.35, edgecolor=GOAL, lw=1.5, zorder=1))
    ax.text(L + 1.5, 1.5, "G", ha="center", va="center", fontsize=13, color=INK, weight="bold")

    ax.add_patch(Rectangle((1, 2), 1, 1, facecolor="none", edgecolor=SW, lw=2, ls="--", zorder=2))
    ax.text(1.5, 2.5, "\u25A3", ha="center", va="center", fontsize=15, color=SW)
    ax.text(1.5, 2.82, "switch", ha="center", va="bottom", fontsize=6.5, color=SW)

    ax.add_patch(Circle((2.5, 0.5), 0.32, facecolor=ROCK, edgecolor="black", lw=1, zorder=3))
    ax.text(2.5, 0.12, "rock", ha="center", va="top", fontsize=6.5, color=ROCK)

    ax.add_patch(Circle((0.5, 1.5), 0.22, facecolor=INK, edgecolor="black", lw=1, zorder=4))
    ax.text(0.5, 1.12, "focal", ha="center", va="top", fontsize=6.5, color=INK)

    ax.add_patch(Circle((1.5, 2.28), 0.16, facecolor=TOP, edgecolor="black", lw=1, zorder=5))
    ax.text(2.9, 2.5, "peer holds switch \u2192 top lane open", fontsize=6.4, color=TOP, va="center")
    ax.text(2.9, 0.5, "peer clears rock \u2192 bottom lane open", fontsize=6.4, color=BOT, va="center")

    ax.annotate("", xy=(L + 1, 1.9), xytext=(1, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=TOP, lw=1.4,
                                connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(L + 1, 1.1), xytext=(1, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=BOT, lw=1.4,
                                connectionstyle="arc3,rad=0.1"))

    ax.text((L + 2) / 2, -0.35, "rendered instance (L=3): mutually exclusive top/bottom lanes",
            ha="center", fontsize=6.8, color=INK)
    fig.savefig(OUT + "fig_env_render.pdf"); plt.close(fig)


if __name__ == "__main__":
    schematic()
    gridworld()
    print("env figures done")
