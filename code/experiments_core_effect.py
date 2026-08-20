"""Core-effect statistics (theory-validation, not performance claims).

(A) Pruning power: fraction of the length-H string space eliminated by the
    soundness constraint (the core never removes an optimum but removes everything
    not containing all core symbols). Surviving fraction shrinks exponentially in L.
(B) Frontier-string length: in SRC the delta=0 frontier is a SINGLE string, so its
    set cardinality |Core| is 1. What grows as Theta(L) is the LENGTH of that string
    (equivalently the number of length-1 patterns it generates). Panel (b) plots that.
(C) Reuse value: a core-guided option chain vs a core-blind random-admissible
    controller under the SAME favourable world; validates that the core carries the
    reusable structure. 10 seeds.

Panels are lettered (a),(b),(c) to match the composite figure in the paper, which
places fig_core_prune.pdf (a,b) beside fig_core_reuse.pdf (c).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src_scaling as S
from plot_style import COLORS, apply_style, grid, ci95_from_std

SEEDS = list(range(10))
OUT = "../figs/"
apply_style()
C = {"core": COLORS["core"], "blind": COLORS["risk"], "ink": COLORS["ink"],
     "blue": COLORS["baseline"]}
res = {}


def pruning_power():
    Ls = [2, 3, 4, 5, 6]
    surviving = []
    for L in Ls:
        env = S.SRC(L=L, N=1)
        core = [env.S0] + env.TOP + [env.G]   # ordered top bottleneck chain
        A = env.NF
        H = L + 2

        def count_contains(H, A, pattern):
            dp = [0] * (len(pattern) + 1); dp[0] = 1
            for _ in range(H):
                ndp = dp[:]
                for j in range(len(pattern)):
                    ndp[j + 1] += dp[j] * 1
                    ndp[j] += dp[j] * (A - 1)
                ndp[len(pattern)] += dp[len(pattern)] * A
                dp = ndp
            return dp[len(pattern)]
        contains = count_contains(H, A, core)
        surviving.append(contains / (A ** H))
    surviving = np.array(surviving)

    # frontier-string LENGTH (not set cardinality: the frontier here is a single string)
    frontier_len = [L + 2 for L in Ls]

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
    ax[0].semilogy(Ls, surviving, marker="o", ms=4, color=C["core"])
    ax[0].set_xlabel("corridor length $L$"); ax[0].set_ylabel("surviving fraction (log)")
    ax[0].set_title("(a) Core pruning is exponential", loc="left")
    ax[0].text(0.04, 0.08, "exact count", transform=ax[0].transAxes,
               fontsize=7.1, color=COLORS["grey"])
    grid(ax[0])
    ax[1].plot(Ls, frontier_len, marker="s", ms=4, color=C["blue"])
    ax[1].set_xlabel("corridor length $L$")
    ax[1].set_ylabel("frontier-string length")
    ax[1].set_title(r"(b) Frontier length is $\Theta(L)$", loc="left")
    ax[1].text(0.04, 0.91, "exact construction", transform=ax[1].transAxes,
               va="top", fontsize=7.1, color=COLORS["grey"])
    grid(ax[1])
    fig.savefig(OUT + "fig_core_prune.pdf"); plt.close(fig)
    res["pruning"] = {"Ls": Ls, "surviving_fraction": surviving.tolist(),
                      "frontier_string_length": frontier_len,
                      "frontier_cardinality": [1 for _ in Ls],
                      "log_slope": float(np.polyfit(Ls, np.log(surviving), 1)[0])}
    print("pruning:", json.dumps(res["pruning"]))


def reuse_value():
    Ls = [2, 3, 4, 5, 6]
    H_of = lambda L: 3 * L + 4
    guided_mean, guided_std, blind_mean, blind_std = [], [], [], []
    for L in Ls:
        env = S.SRC(L=L, N=1); H = H_of(L)
        guided = np.zeros((env.NZ, 4))
        for f in range(env.NF):
            for rock in range(2):
                z = env.zidx(f, rock)
                if f == env.S0: guided[z, 0] = 1.0
                elif f == env.G: guided[z, 3] = 1.0
                else: guided[z, 2] = 1.0
        blind = np.full((env.NZ, 4), 0.25)
        g_s, b_s = [], []
        for s in SEEDS:
            rng = np.random.default_rng(100 * s + L)
            _, scg = env.rollout(guided, 0.0, 5000, H, rng)
            _, scb = env.rollout(blind, 0.0, 5000, H, rng)
            g_s.append(scg.mean()); b_s.append(scb.mean())
        guided_mean.append(np.mean(g_s)); guided_std.append(np.std(g_s))
        blind_mean.append(np.mean(b_s)); blind_std.append(np.std(b_s))

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    Ls_a = np.array(Ls)
    ax.errorbar(Ls_a, guided_mean, yerr=ci95_from_std(guided_std, len(SEEDS)), marker="o",
                color=C["core"], label="core-guided chain")
    ax.errorbar(Ls_a, blind_mean, yerr=ci95_from_std(blind_std, len(SEEDS)), marker="s",
                color=C["blind"], label="core-blind random")
    ax.set_xlabel("corridor length $L$"); ax.set_ylabel("success probability")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(c) Core enables reliable reuse", loc="left")
    ax.legend(title=r"mean $\pm$ 95% CI, 10 seeds", title_fontsize=6.5)
    grid(ax)
    fig.savefig(OUT + "fig_core_reuse.pdf"); plt.close(fig)
    res["reuse"] = {"Ls": Ls, "guided_mean": guided_mean, "guided_std": guided_std,
                    "blind_mean": blind_mean, "blind_std": blind_std}
    print("reuse:", json.dumps(res["reuse"]))


if __name__ == "__main__":
    pruning_power()
    reuse_value()
    json.dump(res, open(OUT + "results_core_effect.json", "w"), indent=2)
    print("core-effect done")
