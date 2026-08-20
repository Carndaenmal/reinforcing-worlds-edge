"""Direct quantitative validations of theorem-level predictions (E7-E9).

E7   Theorem 4.1: sample complexity of delta-family recovery scales as 1/(p0 gamma^2).
E8   Theorem 5.1: tightness of the factor-one bound AT THE TRAJECTORY-LAW LEVEL
     (eps = TV(mu_e, mu_0)).
E9   Theorem 5.1: horizon-uniform tightness IN THE KERNEL PARAMETERIZATION, via an
     explicit depth-indexed cascade family.

Run:  python3 experiments_theory_tests.py       (~5 minutes on one CPU core)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src
from plot_style import COLORS, apply_style, grid

OUT = "../figs/"
H = 12
apply_style()
C = {"a": COLORS["core"], "b": COLORS["baseline"], "c": COLORS["risk"],
     "d": COLORS["orange"], "ink": COLORS["ink"]}
res = {}


# ------------------------------------------------------------------ E7
def _verify_binomial_reduction(pf, alpha, rng, N=4000, reps=300):
    """The plug-in estimator depends on the rollouts only through
    (N_S, #successes containing u), and rollouts are i.i.d., so the exact sampling law is
    N_S ~ Bin(N, p0) and  #occurrences | N_S ~ Bin(N_S, c). This checks that reduction
    against real MDP rollouts before we use it for the (much larger) sweep."""
    pi1 = src.probe_policy(p_fwd=pf); pi2 = src.pi2_alpha(alpha)
    tr, sc = src.rollout(pi1, pi2, 300000, H, rng)
    p0 = float(sc.mean()); m, _ = src.landmark_mass(tr, sc); c = float(m[1])
    emp = []
    for _ in range(reps):
        t2, s2 = src.rollout(pi1, pi2, N, H, rng)
        if s2.sum() == 0: continue
        mm, _ = src.landmark_mass(t2, s2); emp.append(float(mm[1]))
    sim = []
    for _ in range(reps):
        ns = rng.binomial(N, p0)
        if ns == 0: continue
        sim.append(rng.binomial(ns, c) / ns)
    emp = np.array(emp); sim = np.array(sim)
    return {"p0": p0, "c": c, "mdp_mean": float(emp.mean()), "sim_mean": float(sim.mean()),
            "mdp_std": float(emp.std()), "sim_std": float(sim.std())}


def e7_sample_complexity(seed=0):
    """Theorem 4.1: find the smallest N at which the plug-in rule classifies membership
    with error <= 5%, over a grid of (p0, gamma), and test N* ~ 1/(p0 gamma^2)."""
    rng = np.random.default_rng(seed)
    alpha_fixed = 0.05
    gammas = [0.08, 0.05, 0.03]
    REPS = 4000
    Ns = np.unique(np.geomspace(20, 400000, 60).astype(int))

    check = _verify_binomial_reduction(0.9, alpha_fixed, rng)
    print("  reduction check (MDP vs exact sampling law):", json.dumps(check))
    res["e7_reduction_check"] = check

    # Anchor the true coverage on the MDP, then sweep p0 over a wide range inside the
    # (verified) exact sampling law. Varying the probe alone moves p0 only slightly, so
    # sweeping it directly is the only way to test the 1/p0 half of the rate.
    pi1 = src.probe_policy(p_fwd=0.9); pi2 = src.pi2_alpha(alpha_fixed)
    tr, sc = src.rollout(pi1, pi2, 400000, H, rng)
    c_true = float(src.landmark_mass(tr, sc)[0][1])
    p0_grid = [0.8, 0.4, 0.2, 0.1, 0.05]
    rows = []
    for ip, p0 in enumerate(p0_grid):
        for ig, g in enumerate(gammas):
            thresh = c_true - g
            Nstar = None
            for N in Ns:
                ns = rng.binomial(N, p0, size=REPS)
                ok = ns > 0
                chat = np.full(REPS, -1.0)
                chat[ok] = rng.binomial(ns[ok], c_true) / ns[ok]
                if float(np.mean(chat < thresh)) <= 0.05:
                    Nstar = int(N)
                    break

            # Use a separate stream for the Monte Carlo interval so uncertainty
            # estimation cannot alter the frozen point-estimate sweep above.
            rng_ci = np.random.default_rng(20_000 + 100*seed + 10*ip + ig)
            Nlo = None; Nhi = None
            for N in Ns:
                ns = rng_ci.binomial(N, p0, size=REPS)
                ok = ns > 0
                chat = np.full(REPS, -1.0)
                chat[ok] = rng_ci.binomial(ns[ok], c_true) / ns[ok]
                phat = float(np.mean(chat < thresh))
                z = 1.96
                center = (phat + z*z/(2*REPS)) / (1 + z*z/REPS)
                half = z*np.sqrt(phat*(1-phat)/REPS + z*z/(4*REPS*REPS)) / (1 + z*z/REPS)
                lower, upper = center-half, center+half
                if Nlo is None and lower <= 0.05:
                    Nlo = int(N)
                if Nstar is None and phat <= 0.05:
                    Nstar = int(N)
                if Nhi is None and upper <= 0.05:
                    Nhi = int(N)
                if Nlo is not None and Nhi is not None:
                    break
            Nlo = min(Nlo, Nstar); Nhi = max(Nhi, Nstar)
            rows.append({"p0": p0, "gamma": g, "c_true": c_true, "N_star": Nstar,
                         "N_star_ci95": [Nlo, Nhi],
                         "product": (None if Nstar is None else Nstar * p0 * g * g)})
            print(f"  p0={p0:<5} gamma={g}  N*={Nstar:<8} N* p0 g^2={rows[-1]['product']:.4f}")
    res["e7"] = rows

    ok = [r for r in rows if r["N_star"] is not None]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), constrained_layout=True)
    palette = [C["a"], C["b"], C["c"], C["d"], C["ink"]]
    for i, p0v in enumerate(sorted({r["p0"] for r in ok}, reverse=True)):
        sub = [r for r in ok if r["p0"] == p0v]
        x = [1.0 / (r["p0"] * r["gamma"] ** 2) for r in sub]
        y = np.array([r["N_star"] for r in sub], dtype=float)
        lo = np.array([r["N_star_ci95"][0] for r in sub], dtype=float)
        hi = np.array([r["N_star_ci95"][1] for r in sub], dtype=float)
        ax[0].errorbar(x, y, yerr=[np.maximum(0.0, y-lo), np.maximum(0.0, hi-y)],
                       marker=["o", "s", "^", "D", "P"][i], ls="none",
                       color=palette[i % 5],
                       label=rf"$p_0={p0v}$")
    xs = [1.0 / (r["p0"] * r["gamma"] ** 2) for r in ok]
    prod = [r["N_star"] * r["p0"] * r["gamma"] ** 2 for r in ok]
    ref = np.geomspace(min(xs), max(xs), 20)
    ax[0].loglog(ref, float(np.median(prod)) * ref, ls=":", color="0.4",
                 label=r"slope 1 ($\propto \frac{1}{p_0\gamma^2}$)")
    ax[0].set_xlabel(r"$\frac{1}{p_0\gamma^2}$")
    ax[0].set_ylabel("$N^\\star$ (5% error)")
    ax[0].set_title("(a) Sample complexity follows the rate", loc="left")
    ax[0].legend(fontsize=6.2, title="95% simulation interval", title_fontsize=6.2)
    grid(ax[0])
    for j, g in enumerate(gammas):
        sub = sorted([r for r in ok if r["gamma"] == g], key=lambda r: r["p0"])
        xx = np.array([r["p0"] for r in sub])
        yy = np.array([r["product"] for r in sub])
        lo = np.array([r["N_star_ci95"][0] * r["p0"] * r["gamma"]**2 for r in sub])
        hi = np.array([r["N_star_ci95"][1] * r["p0"] * r["gamma"]**2 for r in sub])
        ax[1].errorbar(xx, yy,
                       yerr=[np.maximum(0.0, yy-lo), np.maximum(0.0, hi-yy)],
                       marker=["o", "s", "^"][j],
                       color=[C["a"], C["b"], C["d"]][j], label=rf"$\gamma={g}$")
    ax[1].axhline(float(np.median(prod)), ls=":", color="0.4")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("success probability $p_0$"); ax[1].set_ylabel(r"$N^\star p_0\gamma^2$")
    ax[1].set_ylim(0, max(prod) * 1.35)
    ax[1].set_title("(b) Normalized product is stable", loc="left")
    ax[1].legend(fontsize=6.4, title="95% simulation interval", title_fontsize=6.2)
    grid(ax[1])
    fig.savefig(OUT + "fig_samplecomplexity.pdf"); plt.close(fig)
    res["e7_summary"] = {"products": prod,
                         "ratio_max_min": float(max(prod) / min(prod)),
                         "median_product": float(np.median(prod))}
    print("E7 summary:", json.dumps(res["e7_summary"]))


# ------------------------------------------------------------------ E8
def _abstract_dist(tr, sc):
    """Empirical distribution over abstract outcome classes (top-success, bottom-success, fail)."""
    from collections import Counter
    cnt = Counter()
    for i in range(len(sc)):
        row = tr[i]
        seen = set(int(x) for x in row if x >= 0)
        if not sc[i]:
            cnt["fail"] += 1
        elif 1 in seen:
            cnt["top"] += 1
        else:
            cnt["bot"] += 1
    n = len(sc)
    return {k: v / n for k, v in cnt.items()}


def e8_tightness(seed=1):
    """Theorem 5.1 at the TRAJECTORY-LAW level. Measure the actual TV(mu_alpha, mu_0) and
    compare the realized coverage drop with the guaranteed maximum drop TV/p0."""
    rng = np.random.default_rng(seed)
    pi1 = src.probe_policy()
    NB = 400000
    tr0, sc0 = src.rollout(pi1, src.pi2_alpha(0.0), NB, H, rng)
    p0 = float(sc0.mean()); d0 = _abstract_dist(tr0, sc0)
    m0, _ = src.landmark_mass(tr0, sc0); c0 = float(m0[1])
    n0_success = max(1, int(sc0.sum()))
    c0_se = np.sqrt(max(c0 * (1.0 - c0), 0.0) / n0_success)
    alphas = np.geomspace(1e-3, 0.25, 14)
    tvs, drops, drop_half, ratios, ratio_half = [], [], [], [], []
    for a in alphas:
        tr, sc = src.rollout(pi1, src.pi2_alpha(a), NB, H, rng)
        d = _abstract_dist(tr, sc)
        keys = set(d0) | set(d)
        tv = 0.5 * sum(abs(d.get(k, 0.0) - d0.get(k, 0.0)) for k in keys)
        m, _ = src.landmark_mass(tr, sc); c = float(m[1])
        drop = c0 - c
        n_success = max(1, int(sc.sum()))
        c_se = np.sqrt(max(c * (1.0 - c), 0.0) / n_success)
        dh = 1.96 * np.sqrt(c0_se**2 + c_se**2)
        guar = tv / p0
        tvs.append(tv); drops.append(drop)
        drop_half.append(dh)
        ratios.append(drop / guar if guar > 1e-12 else np.nan)
        ratio_half.append(dh / guar if guar > 1e-12 else np.nan)
    ratios = np.array(ratios)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), constrained_layout=True)
    ax[0].errorbar(tvs, drops, yerr=drop_half, marker="o", color=C["c"],
                   label="realized drop (95% CI)")
    ax[0].plot(tvs, np.array(tvs) / p0, ls="--", color=C["ink"],
               label=r"guaranteed max drop $\frac{\mathrm{TV}}{p_0}$")
    ax[0].set_xlabel(r"$\mathrm{TV}(\mu_\alpha,\mu_0)$"); ax[0].set_ylabel("coverage drop")
    ax[0].set_title("(a) Realized drop reaches the bound", loc="left")
    ax[0].legend(fontsize=6.3, title="400,000 rollouts / point", title_fontsize=6.2)
    grid(ax[0])
    ax[1].errorbar(alphas, ratios, yerr=ratio_half, marker="s", color=C["b"])
    ax[1].axhline(1.0, ls=":", color="0.4")
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.15)
    ax[1].set_xlabel(r"$\alpha$"); ax[1].set_ylabel(r"$\frac{p_0\,\mathrm{drop}}{\mathrm{TV}}$")
    ax[1].set_title("(b) Ratio approaches 1", loc="left")
    ax[1].text(0.04, 0.08, "bars: approximate 95% MC CI", transform=ax[1].transAxes,
               fontsize=6.5, color=COLORS["grey"])
    grid(ax[1])
    fig.savefig(OUT + "fig_tightness.pdf"); plt.close(fig)
    res["e8"] = {"p0": p0, "c0": c0, "alphas": alphas.tolist(), "tv": tvs,
                 "drop": drops, "ratio": ratios.tolist(),
                 "max_ratio": float(np.nanmax(ratios)),
                 "all_le_one": bool(np.all(ratios <= 1.0 + 1e-6))}
    print("E8:", json.dumps({k: res["e8"][k] for k in ["p0", "c0", "max_ratio", "all_le_one"]}))


# ------------------------------------------------------------------ E9
class Cascade:
    """Explicit finite two-player family approaching Theorem 5.1's kernel coefficient.

    Focal positions t_0..t_n (top), b_1..b_n (bypass), goal g. The focal agent has a single
    action, so the probe is trivially fixed. At t_0..t_{n-1} the peer either holds (stay
    on top) or clears (divert to the bypass); t_n then enters g. BOTH routes reach g, so
    p_0 = 1 and diverted mass lands on successes that lack u = (t_1,...,t_n).

    The top trajectory has labels (t_0,...,t_n,g_Sigma), so H = n+2 and the theorem's
    kernel guarantee is (n+1)*alpha. The realized drop has infinitesimal-drift ratio
    n/(n+1), which approaches one across the depth-indexed family.
    """

    def __init__(self, n):
        self.n = n
        self.H = n + 2

    def kernel_dist(self, alpha):
        """At each top state the next-state law moves mass alpha from t_{i+1} to b_{i+1},
        an L1 difference of 2*alpha; the sup over states is the same."""
        return 2.0 * alpha

    def eps(self, alpha):
        return 0.5 * (self.H - 1) * self.kernel_dist(alpha)

    def coverage_exact(self, alpha):
        return (1.0 - alpha) ** self.n

    def rollout(self, alpha, N, rng):
        diverts = rng.random((N, self.n)) < alpha
        return ~diverts.any(axis=1), np.ones(N, dtype=bool)


def e9_kernel_tightness(n=11, alphas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
                        depths=(1, 2, 4, 8, 16, 32, 64, 128, 255),
                        N=4_000_000, seed=7):
    """The kernel coefficient is unimprovable uniformly over horizons.

    Bernoulli's inequality gives realized drop <= n*alpha < (n+1)*alpha = eps/p_0.
    At fixed depth the ratio tends to n/(n+1) as alpha -> 0, and these limits tend to
    one as n grows.
    """
    rng = np.random.default_rng(seed)
    env = Cascade(n)
    rows = []
    for a in alphas:
        u, sc = env.rollout(a, N, rng)
        p0 = float(sc.mean())
        c_mc = float(u[sc].mean())
        drop_mc = 1.0 - c_mc
        se = float(np.sqrt(max(drop_mc, 1e-12) * (1 - drop_mc) / N))   # binomial s.e.
        drop_exact = 1.0 - env.coverage_exact(a)
        guar = env.eps(a) / p0
        rows.append({"alpha": a, "p0": p0, "drop_mc": drop_mc, "drop_mc_se": se,
                     "drop_exact": drop_exact, "eps_over_p0": guar,
                     "ratio_mc": drop_mc / guar, "ratio_exact": drop_exact / guar})
        print(f"  alpha={a:<7} drop={drop_mc:.6f}+-{se:.1e} exact={drop_exact:.6f} "
              f"eps/p0={guar:.6f}  ratio(exact)={drop_exact/guar:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), constrained_layout=True)
    al = [r["alpha"] for r in rows]
    ax[0].errorbar(al, [r["drop_mc"] for r in rows],
                   yerr=[1.96*r["drop_mc_se"] for r in rows],
                   marker="o", ls="none", mfc="white", mec=C["c"], mew=1.0,
                   color=C["c"],
                   label="Monte Carlo (95% CI)")
    ax[0].plot(al, [r["drop_exact"] for r in rows], ls="-", lw=1.0, color=C["a"],
               label=r"realized (exact) $1-(1-\alpha)^n$")
    ax[0].plot(al, [r["eps_over_p0"] for r in rows], ls="--", color=C["ink"],
               label=r"guarantee $\frac{\varepsilon}{p_0}=(n+1)\alpha$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"peer divert probability $\alpha$"); ax[0].set_ylabel("coverage drop")
    ax[0].set_title("(a) Cascade reaches the guarantee", loc="left")
    ax[0].legend(fontsize=6.1, title="4,000,000 rollouts / point", title_fontsize=6.0)
    grid(ax[0])
    depths = np.asarray(depths, dtype=int)
    depth_limits = depths / (depths + 1.0)
    ax[1].plot(depths, depth_limits, marker="s", ms=3.5, color=C["b"],
               label=r"$\lim_{\alpha\to0}\frac{p_0\,\mathrm{drop}}{\varepsilon}$")
    ax[1].axhline(1.0, ls=":", color="0.4")
    ax[1].set_xscale("log", base=2); ax[1].set_ylim(0.45, 1.03)
    ax[1].set_xlabel(r"cascade depth $n$"); ax[1].set_ylabel(r"$\frac{n}{n+1}$")
    ax[1].set_title("(b) Uniform coefficient approaches 1", loc="left")
    ax[1].legend(fontsize=6.5); grid(ax[1])
    fig.savefig(OUT + "fig_kerneltight.pdf"); plt.close(fig)

    res["e9"] = {"n": n, "rows": rows,
                 "max_ratio_exact": float(max(r["ratio_exact"] for r in rows)),
                 "ratio_exact_at_smallest_alpha": float(rows[0]["ratio_exact"]),
                 "small_alpha_limit_fixed_n": float(n / (n + 1.0)),
                 "depths": depths.tolist(),
                 "depth_limits": depth_limits.tolist(),
                 "max_depth_limit": float(depth_limits[-1]),
                 "all_ratios_le_one": bool(all(r["ratio_exact"] <= 1 + 1e-12 for r in rows))}
    print("E9:", json.dumps({k: res["e9"][k] for k in
                             ["max_ratio_exact", "ratio_exact_at_smallest_alpha",
                              "small_alpha_limit_fixed_n", "max_depth_limit",
                              "all_ratios_le_one"]}))


if __name__ == "__main__":
    print("=== E7: sample complexity ===");  e7_sample_complexity()
    print("=== E8: Thm 5.1 tightness (trajectory-law level) ===");  e8_tightness()
    print("=== E9: kernel-level tightness ==="); e9_kernel_tightness()
    json.dump(res, open(OUT + "results_theory_tests.json", "w"), indent=2)
    print("theory tests done")
