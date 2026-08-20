"""Experiments for the invariant-core / boundary-drift paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src
from plot_style import COLORS, apply_style, grid

H = 12
NPROBE = 40000
OUT = "../figs/"
rng = np.random.default_rng(7)

apply_style(9.0)
C = {"d0": COLORS["ink"], "d1": COLORS["baseline"], "d2": COLORS["orange"],
     "d3": COLORS["risk"], "grey": COLORS["grey"], "rob": COLORS["core"]}

results = {}
pi1 = src.probe_policy()


# =============================================================== E1: controlled ramp
def e1():
    alphas = np.concatenate([np.array([0.0]), np.geomspace(2e-4, 0.6, 60)])
    deltas = [0.0, 0.02, 0.10, 0.30]
    surv = {d: [] for d in deltas}
    mass_t1, psucc = [], []
    for a in alphas:
        tr, sc = src.rollout(pi1, src.pi2_alpha(a), NPROBE, H, rng)
        m, _ = src.landmark_mass(tr, sc)
        mass_t1.append(m[1]); psucc.append(sc.mean())
        for d in deltas:
            L = src.delta_landmarks(m, d)
            surv[d].append(1 if 1 in L else 0)   # is t1 still a landmark?
    mass_t1 = np.array(mass_t1)

    first_loss = {}
    for d in deltas:
        s = np.array(surv[d]); idx = np.where(s == 0)[0]
        first_loss[str(d)] = float(alphas[idx[0]]) if len(idx) else None

    effective_successes = np.maximum(1.0, NPROBE * np.asarray(psucc))
    se = np.sqrt(np.clip(mass_t1 * (1.0 - mass_t1), 0.0, None) / effective_successes)
    lower = np.clip(mass_t1 - 1.96 * se, 0.0, 1.0)
    upper = np.clip(mass_t1 + 1.96 * se, 0.0, 1.0)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.45), layout="constrained")
    ax[0].fill_between(alphas, lower, upper, color=C["rob"], alpha=0.16, lw=0)
    ax[0].plot(
        alphas,
        mass_t1,
        color=C["rob"],
        label="40k-rollout coverage (95% CI)",
    )
    for d, c in zip(deltas[1:], [C["d1"], C["d2"], C["d3"]]):
        ax[0].axhline(1 - d, ls=":", lw=0.9, color=c)
        ax[0].text(
            0.98,
            1 - d,
            rf"$\delta={d:g}$",
            transform=ax[0].get_yaxis_transform(),
            color=c,
            fontsize=7.0,
            va="bottom",
            ha="right",
        )
    ax[0].set_xscale("symlog", linthresh=2e-4)
    ax[0].set_xlim(0, 0.8)
    ax[0].set_xlabel(r"peer drift parameter $\alpha$ (0 $=$ no drift)")
    ax[0].set_ylabel("coverage of prototype $t_1$")
    ax[0].set_title(r"(a) Coverage erodes continuously", loc="left")
    ax[0].legend(loc="lower left")
    grid(ax[0], "y")

    palette = [C["d0"], C["d1"], C["d2"], C["d3"]]
    for y, (d, color) in enumerate(zip(deltas, palette)):
        loss = first_loss[str(d)]
        ax[1].hlines(y, 0.0, loss, color=color, linewidth=3.0)
        ax[1].plot(loss, y, marker="|", ms=10, mew=1.5, color=color)
        ax[1].annotate(
            rf"$\alpha={loss:.1g}$",
            xy=(loss, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.0,
            color=COLORS["ink"],
        )
    ax[1].set_xscale("symlog", linthresh=2e-4)
    ax[1].set_xlim(0, 0.8)
    ax[1].set_yticks(range(len(deltas)), [rf"$\delta={d:g}$" for d in deltas])
    ax[1].set_xlabel(r"drift tolerated before $t_1$ exits")
    ax[1].set_title(r"(b) Relaxation delays first exit", loc="left")
    grid(ax[1], "x")
    fig.savefig(OUT + "fig_ramp.pdf")
    plt.close(fig)

    results["e1"] = {"first_loss_alpha": first_loss,
                     "mass_t1_at_alpha0": float(mass_t1[0]),
                     "p_succ_alpha0": float(psucc[0])}
    return alphas, mass_t1, psucc


# =============================================================== E2: bound validation
def e2(alphas, mass_t1, psucc):
    """Theorem 5.1 (sharpened): u in H^delta(mu_0) => c_{mu_e}(u) >= 1 - delta - eps/p_0,
    eps = ((H-1)/2)||P_a - P_0||_{1,inf}. Here u = t1, a phi-bottleneck at alpha=0, so delta = 0."""
    P0 = src.induced_kernel(src.pi2_alpha(0.0))
    kd = np.array([src.kernel_dist(src.induced_kernel(src.pi2_alpha(a)), P0) for a in alphas])
    eps = (H - 1) / 2 * kd
    pmin = psucc[0]
    bound = np.clip(1 - eps / pmin, 0, 1)

    mass_t1 = np.asarray(mass_t1)
    effective_successes = np.maximum(1.0, NPROBE * np.asarray(psucc))
    se = np.sqrt(np.clip(mass_t1 * (1.0 - mass_t1), 0.0, None) / effective_successes)
    fig, ax = plt.subplots(figsize=(3.7, 2.55), layout="constrained")
    ax.fill_between(
        alphas,
        np.clip(mass_t1 - 1.96 * se, 0, 1),
        np.clip(mass_t1 + 1.96 * se, 0, 1),
        color=C["rob"],
        alpha=0.16,
        lw=0,
    )
    ax.plot(alphas, mass_t1, color=C["rob"], label="40k-rollout coverage (95% CI)")
    ax.plot(alphas, bound, color=C["d3"], ls="--", label="theoretical lower bound")
    ax.fill_between(alphas, bound, mass_t1, where=(mass_t1 >= bound),
                    color=C["d3"], alpha=0.08, lw=0)
    ax.set_xscale("symlog", linthresh=2e-4); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"peer drift $\alpha$"); ax.set_ylabel("coverage of $t_1$")
    ax.set_title("Observed coverage stays above the bound", loc="left")
    ax.legend(loc="lower left")
    grid(ax, "y")
    fig.savefig(OUT + "fig_bound.pdf"); plt.close(fig)

    viol = int((mass_t1 < bound - 1e-9).sum())
    results["e2"] = {"violations": viol, "n_points": len(alphas),
                     "kernel_dist_slope": float(kd[-1] / alphas[-1]),
                     "median_slack": float(np.median(mass_t1 - bound)),
                     "min_slack": float(np.min(mass_t1 - bound))}


# =============================================================== E3: independent Q-learners
def iql(eta_peer, n_ep, seed, eps=0.10, eta_focal=0.15, anneal=False, measure_every=5,
        nprobe=6000):
    r = np.random.default_rng(seed)
    Q1 = np.zeros((src.NZ, src.NA1)); Q2 = np.zeros((src.NZ, src.NA2))
    log = {"ep": [], "pol_drift": [], "ker_drift": [], "L": [], "mass": [], "p_clear": []}
    prev_pi2 = None

    def greedy_dist(Q, na):
        pi = np.full((src.NZ, na), eps / na)
        pi[np.arange(src.NZ), Q.argmax(axis=1)] += 1 - eps
        return pi

    for ep in range(n_ep):
        lr2 = eta_peer / (1 + ep / 200) if anneal else eta_peer
        f, rock, t = 0, 0, 0
        while t < H - 1 and f != src.GOAL_F:
            z = src.zidx(f, rock)
            a1 = r.integers(src.NA1) if r.random() < eps else int(Q1[z].argmax())
            a2 = r.integers(src.NA2) if r.random() < eps else int(Q2[z].argmax())
            hold, clear = (a2 == 0), (a2 == 1)
            nf = f
            if f == 0:
                nf = 1 if a1 == 0 else (3 if a1 == 1 else 0)
            elif f == 1 and a1 == 2 and hold and rock == 0:
                nf = 2
            elif f == 2 and a1 == 2:
                nf = 5
            elif f == 3 and a1 == 2 and rock == 1:
                nf = 4
            elif f == 4 and a1 == 2:
                nf = 5
            nrock = max(rock, int(clear))
            r1 = (1.0 if nf == src.GOAL_F else 0.0) - 0.02
            r2 = (1.0 if nf == src.GOAL_F else 0.0) + (0.35 if (clear and rock == 0) else 0.0)
            nz = src.zidx(nf, nrock)
            term = (nf == src.GOAL_F)
            Q1[z, a1] += eta_focal * (r1 + (0 if term else Q1[nz].max()) - Q1[z, a1])
            Q2[z, a2] += lr2 * (r2 + (0 if term else Q2[nz].max()) - Q2[z, a2])
            f, rock, t = nf, nrock, t + 1

        pi2 = greedy_dist(Q2, src.NA2)
        if prev_pi2 is not None and ep % measure_every == 0:
            log["ep"].append(ep)
            log["pol_drift"].append(src.policy_dist(pi2, prev_pi2))
            log["ker_drift"].append(src.kernel_dist(src.induced_kernel(pi2),
                                                    src.induced_kernel(prev_pi2)))
            tr, sc = src.rollout(pi1, pi2, nprobe, H, r)
            m, _ = src.landmark_mass(tr, sc)
            log["mass"].append(m.tolist())
            log["L"].append(sorted(src.delta_landmarks(m, 0.10)))
            log["p_clear"].append(float(pi2[:, 1].mean()))
        if ep % measure_every == 0:
            prev_pi2 = pi2
    return log


def e3():
    log = iql(eta_peer=0.15, n_ep=4000, seed=3)
    ep = np.array(log["ep"]); kd = np.array(log["ker_drift"])
    mass = np.array(log["mass"])
    jac = [src.jaccard(set(log["L"][i]), set(log["L"][i - 1])) for i in range(1, len(log["L"]))]
    cum = np.cumsum(kd)

    fig, ax = plt.subplots(3, 1, figsize=(4.6, 4.4), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1.1]})
    ax[0].plot(ep, kd, color=C["grey"], lw=0.8)
    ax[0].set_ylabel(r"$\|P_e-P_{e-1}\|_{1,\infty}$")
    ax[0].set_title("(a) peer-induced kernel drift is sparse and spiky", loc="left")
    a0 = ax[0].twinx(); a0.plot(ep, cum, color=C["d3"], lw=1.2)
    a0.set_ylabel(r"$V_e$", color=C["d3"]); a0.tick_params(colors=C["d3"]); a0.spines["top"].set_visible(False)

    ax[1].plot(ep, mass[:, 1], color=C["d1"], label=r"$t_1$ (conventional)")
    ax[1].plot(ep, mass[:, 3], color=C["d2"], label=r"$b_1$ (conventional)")
    ax[1].plot(ep, mass[:, 0], color=C["rob"], label=r"$s_0$ (robust)")
    ax[1].axhline(0.9, ls=":", lw=0.8, color="k")
    ax[1].set_ylabel("support mass"); ax[1].legend(loc="center left", ncol=1)
    ax[1].set_title(r"(b) conventional prototypes swap; robust core is fixed", loc="left")

    ax[2].plot(ep[1:], jac, color=C["d0"], lw=0.9)
    ax[2].set_ylabel(r"$J(L_{0.1}(e),L_{0.1}(e{-}1))$"); ax[2].set_xlabel("episode $e$")
    ax[2].set_title("(c) core churn coincides with drift spikes", loc="left")
    fig.savefig(OUT + "fig_iql.pdf"); plt.close(fig)

    jac = np.array(jac)
    kd_al = kd[1:]
    hi = kd_al > np.quantile(kd_al, 0.9)
    results["e3"] = {
        "V_total": float(cum[-1]),
        "frac_episodes_zero_drift": float((kd < 1e-9).mean()),
        "mean_jaccard_high_drift": float(jac[hi].mean()),
        "mean_jaccard_low_drift": float(jac[~hi].mean()),
        "spearman": float(np.corrcoef(np.argsort(np.argsort(kd_al)),
                                      np.argsort(np.argsort(jac)))[0, 1]),
        "robust_mass_min": float(mass[:, 0].min()),
        "t1_mass_start": float(mass[:5, 1].mean()), "t1_mass_end": float(mass[-5:, 1].mean()),
        "b1_mass_start": float(mass[:5, 3].mean()), "b1_mass_end": float(mass[-5:, 3].mean()),
    }


# =============================================================== E4: learning-rate ablation
def e4():
    etas = [0.02, 0.05, 0.10, 0.20, 0.40]
    seeds = [11, 12, 13, 14, 15]
    V_const = np.zeros((len(etas), len(seeds)))
    churn = np.zeros((len(etas), len(seeds)))
    for i, et in enumerate(etas):
        for j, s in enumerate(seeds):
            lg = iql(eta_peer=et, n_ep=2500, seed=s, measure_every=5, nprobe=3000)
            V_const[i, j] = np.sum(lg["ker_drift"])
            Ls = [set(x) for x in lg["L"]]
            churn[i, j] = sum(len(Ls[k] ^ Ls[k - 1]) for k in range(1, len(Ls)))

    V_ann = np.zeros(len(seeds)); ch_ann = np.zeros(len(seeds))
    for j, s in enumerate(seeds):
        lg = iql(eta_peer=0.40, n_ep=2500, seed=s, anneal=True, measure_every=5, nprobe=3000)
        V_ann[j] = np.sum(lg["ker_drift"])
        Ls = [set(x) for x in lg["L"]]
        ch_ann[j] = sum(len(Ls[k] ^ Ls[k - 1]) for k in range(1, len(Ls)))

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.35))
    mV, sV = V_const.mean(1), V_const.std(1)
    ax[0].errorbar(etas, mV, yerr=sV, marker="o", ms=3.5, color=C["d1"], capsize=2, label="constant $\\eta$")
    ax[0].errorbar([0.40], [V_ann.mean()], yerr=[V_ann.std()], marker="s", ms=4.5,
                   color=C["rob"], capsize=2, label=r"annealed $\eta_e=\frac{\eta_0}{1+\frac{e}{200}}$")
    co = np.polyfit(etas, mV, 1)
    ax[0].plot(etas, np.polyval(co, etas), ls=":", color=C["grey"], lw=1)
    ax[0].set_xlabel(r"peer learning rate $\eta$"); ax[0].set_ylabel(r"$V_E$")
    ax[0].set_title(r"(a) $V_E$ grows linearly in $\eta$", loc="left"); ax[0].legend()

    ax[1].errorbar(etas, churn.mean(1), yerr=churn.std(1), marker="o", ms=3.5,
                   color=C["d2"], capsize=2, label="constant $\\eta$")
    ax[1].errorbar([0.40], [ch_ann.mean()], yerr=[ch_ann.std()], marker="s", ms=4.5,
                   color=C["rob"], capsize=2, label="annealed")
    ax[1].set_xlabel(r"peer learning rate $\eta$")
    ax[1].set_ylabel(r"total $L_{0.1}$ churn $\sum_e |L_e \triangle L_{e-1}|$")
    ax[1].set_title("(b) prototype churn follows $V_E$", loc="left"); ax[1].legend()
    fig.savefig(OUT + "fig_ablation.pdf"); plt.close(fig)

    results["e4"] = {
        "etas": etas, "V_mean": mV.tolist(), "V_std": sV.tolist(),
        "churn_mean": churn.mean(1).tolist(), "churn_std": churn.std(1).tolist(),
        "V_anneal_mean": float(V_ann.mean()), "V_anneal_std": float(V_ann.std()),
        "churn_anneal_mean": float(ch_ann.mean()),
        "V_reduction_factor": float(mV[-1] / V_ann.mean()),
        "churn_reduction_factor": float(churn.mean(1)[-1] / max(ch_ann.mean(), 1e-9)),
        "lin_fit_slope": float(co[0]),
        "corr_V_churn": float(np.corrcoef(V_const.ravel(), churn.ravel())[0, 1]),
    }


# =============================================================== E5: membership vs executability
def pi2_beta(beta):
    """Peer holds the switch w.p. beta, idles otherwise; NEVER clears.
    Bottom route stays impossible, so t1,t2 remain landmarks for every beta>0."""
    p = np.zeros((src.NZ, src.NA2)); p[:, 0] = beta; p[:, 2] = 1 - beta
    return p


def chain_success(beta, k, n=40000):
    """Fixed option chain calibrated at beta=1: up; (fwd at t1, <=k attempts); fwd; fwd."""
    r = np.random.default_rng(99)
    succ = np.zeros(n, dtype=bool)
    holds = r.random((n, k)) < beta
    succ = holds.any(axis=1)   # needs at least one hold within its k-step budget
    return succ.mean()


def e5():
    betas = np.linspace(0.05, 1.0, 20)
    memb, exe = [], []
    for b in betas:
        tr, sc = src.rollout(pi1, pi2_beta(b), NPROBE, H, rng)
        m, _ = src.landmark_mass(tr, sc)
        memb.append(1 if {0, 1, 2, 5} <= src.delta_landmarks(m, 0.0) else 0)
        exe.append(chain_success(b, k=2))

    exe = np.asarray(exe)
    exact = 1.0 - (1.0 - betas) ** 2
    se = np.sqrt(np.clip(exe * (1.0 - exe), 0.0, None) / 40_000.0)
    fig, ax = plt.subplots(figsize=(3.7, 2.5), layout="constrained")
    ax.step(
        betas,
        memb,
        where="post",
        color=C["d1"],
        label=r"core membership (exact)",
    )
    ax.fill_between(
        betas,
        np.clip(exe - 1.96 * se, 0, 1),
        np.clip(exe + 1.96 * se, 0, 1),
        color=C["d3"],
        alpha=0.16,
        lw=0,
        label="Monte Carlo 95% CI",
    )
    ax.plot(betas, exact, color=C["d3"], label="option success (exact)")
    ax.scatter(betas, exe, s=8, facecolors="white", edgecolors=C["d3"], linewidths=0.7)
    ax.set_xlabel(r"peer hold probability $\beta$"); ax.set_ylabel("probability")
    ax.set_ylim(-0.05, 1.08); ax.invert_xaxis()
    ax.set_title("Membership persists while execution fails", loc="left")
    ax.legend(loc="lower right")
    grid(ax, "y")
    fig.savefig(OUT + "fig_exec.pdf"); plt.close(fig)
    results["e5"] = {"membership_always_held": bool(all(memb)),
                     "chain_success_at_beta_1": float(exe[-1]),
                     "chain_success_at_beta_0.05": float(exe[0])}


if __name__ == "__main__":
    a, m, p = e1(); print("E1 done", results["e1"])
    e2(a, m, p); print("E2 done", results["e2"])
    e3(); print("E3 done", results["e3"])
    e5(); print("E5 done", results["e5"])
    e4(); print("E4 done", results["e4"])
    with open("../figs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nALL RESULTS\n", json.dumps(results, indent=2))
