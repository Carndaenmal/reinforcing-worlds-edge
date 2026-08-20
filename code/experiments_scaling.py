"""Scaling experiments (theory-validation).

HEADLINE: boundary drift destroys the invariant core *faster and more severely*
as either the number of agents N or the environment size L grows. 10 seeds.
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src_scaling as S
from plot_style import COLORS, apply_style, grid

SEEDS = list(range(10))
OUT = "../figs/"
apply_style()
C = [COLORS["ink"], COLORS["baseline"], COLORS["orange"], COLORS["risk"],
     COLORS["core"], COLORS["grey"], "#7B2CBF"]
res = {}
H_of = lambda L: 3 * L + 4


def destruction_and_drift():
    Ls = [2, 3, 4, 5, 6, 8]
    Ns = [1, 2, 3, 4, 6, 8]
    eta = 0.0002        # fine per-peer step so time-to-destroy is well resolved
    n_upd = 3000
    fragility = np.zeros((len(Ls), len(Ns)))   # 1000 / E*(full-core destruction)
    fragility_half = np.zeros_like(fragility)  # pairing-bootstrap 95% CI half-width
    VE = np.zeros((len(Ls), len(Ns)))          # aggregate boundary movement at fixed budget
    for il, L in enumerate(Ls):
        for iN, N in enumerate(Ns):
            env = S.SRC(L=L, N=N); pi1 = env.probe_policy(); H = H_of(L)
            core_syms = env.TOP[:]
            surv_runs = []
            for s in SEEDS:
                rng = np.random.default_rng(31 * s + 7 * il + iN)
                alpha = 0.0; surv = n_upd
                for k in range(1, n_upd + 1):
                    alpha = min(1.0, alpha + eta)
                    q = S.q_of_alpha(alpha, N)
                    tr, sc = env.rollout(pi1, q, 3000, H, rng)
                    m, _ = env.landmark_mass(tr, sc)
                    if all(c not in env.delta_core(m, 0.05) for c in core_syms):
                        surv = k; break
                surv_runs.append(surv)
            fragility[il, iN] = 1000.0 / np.mean(surv_runs)
            boot_rng = np.random.default_rng(9000 + 101 * il + iN)
            surv_arr = np.asarray(surv_runs, dtype=float)
            draws = boot_rng.integers(0, len(surv_arr), size=(10_000, len(surv_arr)))
            frag_draws = 1000.0 / surv_arr[draws].mean(axis=1)
            lo, hi = np.quantile(frag_draws, [0.025, 0.975])
            fragility_half[il, iN] = max(fragility[il, iN] - lo,
                                         hi - fragility[il, iN])
            # Aggregate boundary movement at a fixed budget: total L1 kernel mass
            # displaced after n_budget peer steps. Sup-norm drift depends only on N,
            # but the AGGREGATE boundary movement grows with L too (a longer corridor
            # has more perturbed transitions), so V_E^agg grows in both N and L.
            n_budget = 10; eta_b = 0.02
            alpha = 0.0; V = 0.0
            Pprev = env.induced_kernel(0.0)
            for _ in range(n_budget):
                alpha = min(1.0, alpha + eta_b)
                q = S.q_of_alpha(alpha, N)
                Pcur = env.induced_kernel(q)
                V += np.abs(Pcur - Pprev).sum()   # aggregate L1 movement
                Pprev = Pcur
            VE[il, iN] = V

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    im1 = ax1.imshow(fragility, origin="lower", aspect="auto", cmap="magma")
    ax1.set_xticks(range(len(Ns)), Ns); ax1.set_yticks(range(len(Ls)), Ls)
    ax1.set_xlabel("number of peers $N$"); ax1.set_ylabel("corridor length $L$")
    ax1.set_title(r"(a) Core fragility $\frac{10^3}{E^\star}$", loc="left")
    cut1 = 0.55 * (fragility.min() + fragility.max())
    for il in range(len(Ls)):
        for iN in range(len(Ns)):
            ax1.text(iN, il, f"{fragility[il, iN]:.2f}\n$\\pm${fragility_half[il, iN]:.2f}",
                     ha="center", va="center", fontsize=5.4,
                     color="white" if fragility[il, iN] < cut1 else "black")
    fig.colorbar(im1, ax=ax1, fraction=0.046, label=r"$\frac{10^3}{E^\star}$")
    ax1.text(0.01, -0.29, r"cell: estimate $\pm$ 95% seed-bootstrap half-width",
             transform=ax1.transAxes, fontsize=6.2, color=COLORS["grey"])

    im2 = ax2.imshow(VE, origin="lower", aspect="auto", cmap="viridis")
    ax2.set_xticks(range(len(Ns)), Ns); ax2.set_yticks(range(len(Ls)), Ls)
    ax2.set_xlabel("number of peers $N$"); ax2.set_ylabel("corridor length $L$")
    ax2.set_title(r"(b) Aggregate drift $V_E^{\mathrm{agg}}$", loc="left")
    cut2 = 0.52 * (VE.min() + VE.max())
    for il in range(len(Ls)):
        for iN in range(len(Ns)):
            ax2.text(iN, il, f"{VE[il, iN]:.1f}", ha="center", va="center", fontsize=5.8,
                     color="white" if VE[il, iN] < cut2 else "black")
    fig.colorbar(im2, ax=ax2, fraction=0.046, label=r"$V_E^{\mathrm{agg}}$")
    ax2.text(0.01, -0.29, "cell: exact accumulated kernel movement",
             transform=ax2.transAxes, fontsize=6.2, color=COLORS["grey"])
    fig.savefig(OUT + "fig_scale_surface.pdf"); plt.close(fig)

    res["surface"] = {
        "Ls": Ls, "Ns": Ns,
        "fragility_min_max": [float(fragility.min()), float(fragility.max())],
        "VE_min_max": [float(VE.min()), float(VE.max())],
        "fragility_monotone_in_L": bool(np.all(np.diff(fragility, axis=0) >= -1e-6)),
        "fragility_monotone_in_N": bool(np.all(np.diff(fragility, axis=1) >= -1e-6)),
        "VE_monotone_in_L": bool(np.all(np.diff(VE, axis=0) >= -1e-6)),
        "VE_monotone_in_N": bool(np.all(np.diff(VE, axis=1) >= -1e-6)),
        "fragility": fragility.tolist(), "fragility_ci95_half": fragility_half.tolist(),
        "VE": VE.tolist(),
    }
    print("surface:", json.dumps({k: res["surface"][k] for k in
          ["fragility_min_max", "VE_min_max", "fragility_monotone_in_L",
           "fragility_monotone_in_N", "VE_monotone_in_L", "VE_monotone_in_N"]}))


def survival_slices():
    L0 = 3; H = H_of(L0); eta = 0.0002; n_upd = 1500
    Ns = [1, 2, 3, 4, 6, 8]; survN_m, survN_s, survN_ci = [], [], []
    for N in Ns:
        env = S.SRC(L=L0, N=N); pi1 = env.probe_policy(); survs = []
        for s in SEEDS:
            rng = np.random.default_rng(7 * s + N); alpha = 0.0; surv = n_upd
            for k in range(1, n_upd + 1):
                alpha = min(1.0, alpha + eta); q = S.q_of_alpha(alpha, N)
                tr, sc = env.rollout(pi1, q, 4000, H, rng); m, _ = env.landmark_mass(tr, sc)
                if env.TOP[0] not in env.delta_core(m, 0.05): surv = k; break
            survs.append(surv)
        survN_m.append(np.mean(survs)); survN_s.append(np.std(survs))
        survN_ci.append(1.96 * np.std(survs, ddof=1) / np.sqrt(len(SEEDS)))

    N0 = 2; eta = 0.0002; n_upd = 2500
    Ls = [2, 3, 4, 5, 6, 8]; survL_m, survL_s, survL_ci = [], [], []
    for L in Ls:
        env = S.SRC(L=L, N=N0); pi1 = env.probe_policy(); H = H_of(L); survs = []
        core_syms = env.TOP[:]
        for s in SEEDS:
            rng = np.random.default_rng(11 * s + L); alpha = 0.0; surv = n_upd
            for k in range(1, n_upd + 1):
                alpha = min(1.0, alpha + eta); q = S.q_of_alpha(alpha, N0)
                tr, sc = env.rollout(pi1, q, 4000, H, rng); m, _ = env.landmark_mass(tr, sc)
                core_now = env.delta_core(m, 0.05)
                if all(c not in core_now for c in core_syms): surv = k; break
            survs.append(surv)
        survL_m.append(np.mean(survs)); survL_s.append(np.std(survs))
        survL_ci.append(1.96 * np.std(survs, ddof=1) / np.sqrt(len(SEEDS)))

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
    Ns_a = np.array(Ns)
    ax[0].errorbar(Ns_a, survN_m, yerr=survN_ci, marker="o", color=C[1],
                   label=r"mean $\pm$ 95% CI")
    cfit = survN_m[0] * Ns_a[0]
    ax[0].plot(Ns_a, cfit / Ns_a, ls=":", color=C[5], label=r"$\propto \frac{1}{N}$")
    r2N = np.corrcoef(1.0 / Ns_a, survN_m)[0, 1] ** 2
    ax[0].set_xlabel("number of peers $N$"); ax[0].set_ylabel(r"core survival $E^\star$")
    ax[0].set_title(rf"(a) Survival $\propto \frac{{1}}{{N}}$ ($R^2{{=}}{r2N:.3f}$)", loc="left")
    ax[0].legend(title="10 seeds", title_fontsize=6.6); grid(ax[0])
    Ls_a = np.array(Ls)
    ax[1].errorbar(Ls_a, survL_m, yerr=survL_ci, marker="s", color=C[3],
                   label=r"mean $\pm$ 95% CI")
    ax[1].set_xlabel("corridor length $L$"); ax[1].set_ylabel(r"full-core survival $E^\star$")
    ax[1].set_title(r"(b) Longer corridors erode sooner", loc="left")
    ax[1].legend(title="10 seeds", title_fontsize=6.6); grid(ax[1])
    fig.savefig(OUT + "fig_scale_survival.pdf"); plt.close(fig)
    res["slices"] = {"Ns": Ns, "survN_mean": survN_m, "survN_std": survN_s, "R2_1overN": float(r2N),
                     "survN_ci95_half": survN_ci, "Ls": Ls,
                     "survL_mean": survL_m, "survL_std": survL_s,
                     "survL_ci95_half": survL_ci,
                     "survL_decreasing": bool(np.all(np.diff(survL_m) <= 1e-6))}
    print("slices:", json.dumps({"R2_1overN": res["slices"]["R2_1overN"],
                                 "survL_mean": survL_m, "survL_decreasing": res["slices"]["survL_decreasing"]}))


def bound_validity():
    """Figure 11. The Thm 5.1 guarantee is 1 - delta - eps/p0 (delta = 0 for the top
    prototype). Reporting a grid of binary 1.00s is uninformative; we instead report the
    MINIMUM SLACK (empirical coverage minus guarantee) over the drift levels at which the
    guarantee is NON-VACUOUS, i.e. where 1 - eps/p0 > 0. This matters: once eps/p0 >= 1 the
    guarantee clips to 0 and any coverage trivially satisfies it, so slack computed there
    measures nothing. Cells with no non-vacuous drift level are marked 'vac.'."""
    Ls = [2, 3, 4, 6, 8]; Ns = [1, 2, 4, 8]
    grid_valid = np.zeros((len(Ls), len(Ns)))
    grid_slack = np.full((len(Ls), len(Ns)), np.nan)
    grid_nnv = np.zeros((len(Ls), len(Ns)), dtype=int)   # number of non-vacuous levels
    alphas = np.geomspace(1e-3, 0.3, 12)
    for il, L in enumerate(Ls):
        for iN, N in enumerate(Ns):
            env = S.SRC(L=L, N=N); pi1 = env.probe_policy(); H = H_of(L)
            viol = 0; tot = 0; slacks = []
            for a in alphas:
                q = S.q_of_alpha(a, N)
                eps = 0.5 * (H - 1) * S.kernel_dist_q(env, q, 0.0)
                for sd in SEEDS:
                    rng = np.random.default_rng(1000 * sd + il * 37 + iN)
                    tr0, sc0 = env.rollout(pi1, 0.0, 8000, H, rng); p0 = sc0.mean()
                    tr, sc = env.rollout(pi1, q, 8000, H, rng); m, _ = env.landmark_mass(tr, sc)
                    cov = m[env.TOP[0]]
                    raw = 1.0 - eps / max(p0, 1e-9)      # Thm 5.1 with delta = 0
                    pred = max(0.0, raw)
                    tot += 1
                    if cov < pred - 1e-6: viol += 1
                    if raw > 0.0:                         # only meaningful where non-vacuous
                        slacks.append(cov - pred)
            grid_valid[il, iN] = 1 - viol / max(tot, 1)
            grid_nnv[il, iN] = len(slacks)
            if slacks:
                grid_slack[il, iN] = float(np.min(slacks))

    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    vmax = float(np.nanmax(grid_slack)) if np.isfinite(grid_slack).any() else 1.0
    im = ax.imshow(grid_slack, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
    ax.set_yticks(range(len(Ls))); ax.set_yticklabels(Ls)
    ax.set_xlabel("number of peers $N$"); ax.set_ylabel("corridor length $L$")
    ax.set_title("Observed minimum non-vacuous slack", loc="left")
    for il in range(len(Ls)):
        for iN in range(len(Ns)):
            v = grid_slack[il, iN]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((iN - 0.5, il - 0.5), 1, 1, fill=True,
                                           color="0.88", zorder=1))
                ax.text(iN, il, "vac.", ha="center", va="center", fontsize=6.5,
                        color="0.25", zorder=3)
            else:
                ax.text(iN, il, f"{v:.3f}", ha="center", va="center", fontsize=6.5, zorder=3,
                        color="white" if v < 0.5 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="minimum slack")
    ax.text(0.01, -0.25, "minimum over tested drift levels and 10 seeds",
            transform=ax.transAxes, fontsize=6.4, color=COLORS["grey"])
    fig.savefig(OUT + "fig_scale_bound.pdf"); plt.close(fig)
    res["bound_validity"] = {
        "Ls": Ls, "Ns": Ns,
        "violations_fraction_min": float(grid_valid.min()),
        "all_valid": bool((grid_valid >= 0.999).all()),
        "min_slack_nonvacuous": (float(np.nanmin(grid_slack))
                                 if np.isfinite(grid_slack).any() else None),
        "grid_min_slack": grid_slack.tolist(),
        "n_nonvacuous_levels": grid_nnv.tolist()}
    print("bound_validity:", json.dumps(res["bound_validity"]))


def core_cost():
    rng = np.random.default_rng(0)
    Hs = [4, 5, 6, 7, 8, 9]; ks = [2, 3, 4, 5]
    cost = np.full((len(ks), len(Hs)), np.nan)
    cost_raw = np.full((len(ks), len(Hs), 5), np.nan)
    for ik, k in enumerate(ks):
        for ih, H in enumerate(Hs):
            reps = []
            for _ in range(5):
                seqs = [[0] + list(rng.integers(1, 4, size=H - 2)) + [9] for _ in range(k)]
                _, cells = S.lcs_core_bruteforce(seqs); reps.append(cells)
            cost_raw[ik, ih] = reps
            cost[ik, ih] = np.mean(reps)
    cost_ci = 1.96 * np.std(cost_raw, axis=2, ddof=1) / np.sqrt(cost_raw.shape[2])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
    for ik, k in enumerate(ks):
        ax[0].errorbar(Hs, cost[ik], yerr=cost_ci[ik], marker="o", color=C[ik],
                       zorder=3)
        ax[0].annotate(f"$k={k}$", xy=(Hs[-1], cost[ik, -1]), xytext=(5, 0),
                       textcoords="offset points", ha="left", va="center",
                       fontsize=6.8, color=C[ik], clip_on=False)
    ax[0].set_yscale("log"); ax[0].set_xlabel("horizon $H$"); ax[0].set_ylabel("LCS DP cells (log)")
    ax[0].set_title(r"(a) Exact-core cost grows as $O(H^k)$", loc="left")
    ax[0].set_xlim(3.75, 9.65)
    ax[0].text(0.03, 0.97, r"mean $\pm$ 95% CI, 5 instances",
               transform=ax[0].transAxes, ha="left", va="top",
               fontsize=6.3, color=COLORS["grey"])
    grid(ax[0])
    exps = [float(np.polyfit(np.log(Hs), np.log(cost[ik] + 1), 1)[0]) for ik in range(len(ks))]
    boot_rng = np.random.default_rng(17)
    boot_exps = np.zeros((10_000, len(ks)))
    for b in range(len(boot_exps)):
        for ik in range(len(ks)):
            picked = [cost_raw[ik, ih, boot_rng.integers(0, 5)] for ih in range(len(Hs))]
            boot_exps[b, ik] = np.polyfit(np.log(Hs), np.log(np.asarray(picked) + 1), 1)[0]
    exp_lo, exp_hi = np.quantile(boot_exps, [0.025, 0.975], axis=0)
    ax[1].errorbar(ks, exps, yerr=[np.asarray(exps)-exp_lo, exp_hi-np.asarray(exps)],
                   marker="s", color=C[3], label="fitted exponent")
    ax[1].plot(ks, ks, ls=":", color=C[5], label="$y=k$")
    ax[1].set_xlabel("number of sequences $k$"); ax[1].set_ylabel("fitted exponent")
    ax[1].set_title("(b) Exponent tracks $k$", loc="left")
    ax[1].legend(title="95% bootstrap CI", title_fontsize=6.5); grid(ax[1])
    fig.savefig(OUT + "fig_scale_cost.pdf"); plt.close(fig)
    res["core_cost"] = {"Hs": Hs, "ks": ks, "fitted_exponents": exps,
                        "fitted_exponent_ci95": np.column_stack([exp_lo, exp_hi]).tolist()}
    print("core_cost:", json.dumps(exps))


if __name__ == "__main__":
    t = time.time(); destruction_and_drift(); print("  surfaces %.1fs" % (time.time() - t))
    t = time.time(); survival_slices(); print("  slices %.1fs" % (time.time() - t))
    t = time.time(); bound_validity(); print("  validity %.1fs" % (time.time() - t))
    t = time.time(); core_cost(); print("  cost %.1fs" % (time.time() - t))
    json.dump(res, open(OUT + "results_scaling.json", "w"), indent=2)
    print("scaling done")
