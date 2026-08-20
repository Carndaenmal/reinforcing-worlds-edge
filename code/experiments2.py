"""E3: value-based (argmax) peer  -> sparse, large-jump drift.
E4: softmax policy-gradient peer -> exact population and finite-evaluation exits."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src
from plot_style import COLORS, apply_style, grid, mean_ci95

H = 12
OUT = "../figs/"
EPS_MEAS = 0.02          # measurement smoothing of the peer's convention
apply_style()
C = {"d0": COLORS["ink"], "d1": COLORS["baseline"], "d2": COLORS["orange"],
     "d3": COLORS["risk"], "grey": COLORS["grey"], "rob": COLORS["core"]}
pi1 = src.probe_policy()
res = {}


def smooth_greedy(Q, na, eps=EPS_MEAS):
    pi = np.full((src.NZ, na), eps / na)
    pi[np.arange(src.NZ), Q.argmax(axis=1)] += 1 - eps
    return pi


# ============================================================ E3: argmax peer
def run_iql(seed, n_ep=8000, eps=0.20, lr1=0.20, lr2=0.20, meas=25, nprobe=7000):
    """Two independent Q-learners. The peer has NO private reward beyond the shared
    goal, so convention choice (top vs bottom route) is a pure symmetry-breaking
    coordination problem; exploration keeps flipping it during co-adaptation."""
    r = np.random.default_rng(seed)
    Q1 = np.zeros((src.NZ, src.NA1)); Q2 = np.zeros((src.NZ, src.NA2))
    log = {"ep": [], "kd": [], "pd": [], "mass": [], "L": []}
    prev = None
    for ep in range(n_ep):
        f, rock, t = 0, 0, 0
        while t < H - 1 and f != 5:
            z = src.zidx(f, rock)
            a1 = r.integers(4) if r.random() < eps else int(Q1[z].argmax())
            a2 = r.integers(3) if r.random() < eps else int(Q2[z].argmax())
            hold, clear = (a2 == 0), (a2 == 1)
            nf = f
            if f == 0:   nf = 1 if a1 == 0 else (3 if a1 == 1 else 0)
            elif f == 1 and a1 == 2 and hold and rock == 0: nf = 2
            elif f == 2 and a1 == 2: nf = 5
            elif f == 3 and a1 == 2 and rock == 1: nf = 4
            elif f == 4 and a1 == 2: nf = 5
            nrock = max(rock, int(clear)); term = (nf == 5)
            r1 = (1.0 if term else 0.0) - 0.02
            r2 = (1.0 if term else 0.0)
            nz = src.zidx(nf, nrock)
            Q1[z, a1] += lr1 * (r1 + (0 if term else Q1[nz].max()) - Q1[z, a1])
            Q2[z, a2] += lr2 * (r2 + (0 if term else Q2[nz].max()) - Q2[z, a2])
            f, rock, t = nf, nrock, t + 1
        if ep % meas == 0:
            pi2 = smooth_greedy(Q2, src.NA2)
            if prev is not None:
                log["ep"].append(ep)
                log["kd"].append(src.kernel_dist(src.induced_kernel(pi2),
                                                 src.induced_kernel(prev)))
                log["pd"].append(src.policy_dist(pi2, prev))
                tr, sc = src.rollout(pi1, pi2, nprobe, H, r)
                m, _ = src.landmark_mass(tr, sc)
                log["mass"].append(m.tolist())
                log["L"].append(sorted(src.delta_landmarks(m, 0.10)))
            prev = pi2
    return log


def e3():
    logs = [run_iql(s) for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    lg = logs[0]
    ep = np.array(lg["ep"]); kd = np.array(lg["kd"]); mass = np.array(lg["mass"])
    cum = np.cumsum(kd)
    route_changes = [int(np.sum(np.diff(
        np.where(np.array(l["mass"])[:, 1] >= np.array(l["mass"])[:, 3], 0, 1)) != 0))
        for l in logs]
    robust_min = [float(np.array(l["mass"])[:, 0].min()) for l in logs]
    conv_frac = [float((np.array(l["mass"])[:, 1] >= 0.9).mean()) for l in logs]
    route_mean, route_lo, route_hi = mean_ci95(route_changes)

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    ax[0].plot(ep, cum, color=C["d3"], lw=1.7)
    ax[0].fill_between(ep, 0, cum, color=C["d3"], alpha=0.09)
    ax[0].set_xlabel("episode $e$"); ax[0].set_ylabel(r"cumulative drift $V_e$")
    ax[0].set_title("(a) Drift accumulates", loc="left")
    ax[0].text(0.96, 0.08, "representative seed", transform=ax[0].transAxes,
               va="bottom", ha="right", fontsize=7.1, color=C["grey"])
    grid(ax[0])

    ax[1].plot(ep, mass[:, 1], color=C["d1"], lw=1.0, alpha=0.9,
               label=r"top prototype $t_1$")
    ax[1].plot(ep, mass[:, 3], color=C["d2"], lw=1.0, alpha=0.9,
               label=r"bottom prototype $b_1$")
    ax[1].plot(ep, mass[:, 0], color=C["rob"], lw=2.1,
               label=r"robust core $s_0$")
    ax[1].axhline(0.9, ls=":", lw=0.9, color=C["d0"])
    ax[1].text(ep[-1], 0.875, r"$1-\delta$", ha="right", va="top",
               fontsize=7.0, color=C["d0"])
    ax[1].set_xlabel("episode $e$"); ax[1].set_ylabel("successful-path coverage")
    ax[1].set_ylim(-0.03, 1.08)
    ax[1].set_title("(b) Conventions churn; core stays", loc="left")
    ax[1].legend(loc="upper right", bbox_to_anchor=(0.99, 0.84), fontsize=6.6)
    grid(ax[1])

    y = np.arange(1, len(route_changes) + 1)
    ax[2].scatter(route_changes, y, s=22, color=C["d1"], edgecolor="white",
                  linewidth=0.45, zorder=3)
    ax[2].axvspan(route_lo, route_hi, color=C["d3"], alpha=0.13, zorder=1)
    ax[2].axvline(route_mean, color=C["d3"], lw=1.7, zorder=2)
    ax[2].text(0.02, 0.97,
               f"mean {route_mean:.1f}\n95% CI [{route_lo:.1f}, {route_hi:.1f}]",
               transform=ax[2].transAxes, ha="left", va="top",
               fontsize=7.0, color=C["d3"])
    ax[2].set_xlabel("convention flips in 8,000 episodes")
    ax[2].set_ylabel("seed"); ax[2].set_yticks(y)
    ax[2].set_ylim(0.3, len(route_changes) + 1.35)
    ax[2].set_title("(c) Churn repeats across seeds", loc="left")
    grid(ax[2], axis="x")
    fig.savefig(OUT + "fig_iql.pdf"); plt.close(fig)

    res["e3"] = {
        "n_seeds": len(logs),
        "route_changes_per_seed": route_changes,
        "route_changes_mean": float(np.mean(route_changes)),
        "robust_core_min_per_seed": robust_min,
        "robust_core_always_present": bool(all(x >= 0.999 for x in robust_min)),
        "frac_time_conventionA_is_landmark": conv_frac,
        "V_total_per_seed": [float(np.sum(l["kd"])) for l in logs],
    }
    print("E3", json.dumps(res["e3"], indent=2))


# ============================================================ E4: softmax PG peer
def pg_rollout(theta, n, r):
    """Vectorised rollout with fixed focal probe + softmax peer.
    Returns per-step (z, a2) and peer return per trajectory."""
    pi2 = np.exp(theta - theta.max(1, keepdims=True))
    pi2 /= pi2.sum(1, keepdims=True)
    f = np.zeros(n, dtype=np.int64); rock = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool); ret = np.zeros(n)
    Z = np.full((n, H), -1, dtype=np.int64); A = np.full((n, H), -1, dtype=np.int64)
    for t in range(max(H - 1, 0)):
        z = f * 2 + rock
        u = r.random(n); a1 = (u[:, None] > np.cumsum(pi1[z], 1)).sum(1)
        u2 = r.random(n); a2 = (u2[:, None] > np.cumsum(pi2[z], 1)).sum(1)
        Z[:, t] = np.where(done, -1, z); A[:, t] = np.where(done, -1, a2)
        hold, clear = (a2 == 0), (a2 == 1)
        nf = f.copy()
        m = (f == 0); nf[m & (a1 == 0)] = 1; nf[m & (a1 == 1)] = 3
        m = (f == 1); nf[m & (a1 == 2) & hold & (rock == 0)] = 2
        m = (f == 2); nf[m & (a1 == 2)] = 5
        m = (f == 3); nf[m & (a1 == 2) & (rock == 1)] = 4
        m = (f == 4); nf[m & (a1 == 2)] = 5
        first_clear = clear & (rock == 0) & ~done
        ret += 0.35 * first_clear
        nrock = np.maximum(rock, clear.astype(np.int64))
        newly = (~done) & (nf == 5); ret += 1.0 * newly
        f = np.where(done, f, nf); rock = np.where(done, rock, nrock)
        done = done | (f == 5)
    return Z, A, ret, pi2


def run_pg(eta, seed, n_upd=400, batch=192, anneal=False, meas=15, nprobe=5000,
           delta=0.10, theta0=6.0):
    r_train = np.random.default_rng(seed)
    r_eval = np.random.default_rng(seed + 100_000)
    theta = np.zeros((src.NZ, src.NA2)); theta[:, 0] = theta0
    pi2_init = np.exp(theta - theta.max(1, keepdims=True))
    pi2_init /= pi2_init.sum(1, keepdims=True)
    c0, p0 = src.exact_landmark_coverage(pi1, pi2_init, H, symbol=1)
    prev_step = pi2_init
    V = 0.0
    log = {
        "upd": [], "V": [], "mass_t1": [], "population_mass_at_eval": [], "in_L": [],
        "population_upd": [0], "population_mass_t1": [float(c0)],
        "population_success": [float(p0)],
    }
    surv = None
    pop_surv = 0 if c0 < 1.0 - delta else None
    grid_pop_surv = None
    for k in range(n_upd):
        lr = eta / (1 + k / 150) if anneal else eta
        Z, A, ret, pi2 = pg_rollout(theta, batch, r_train)
        adv = ret - ret.mean()
        g = np.zeros_like(theta)
        valid = Z >= 0
        fz = Z[valid]; fa = A[valid]
        fadv = np.broadcast_to(adv[:, None], Z.shape)[valid]
        np.add.at(g, (fz, fa), fadv)
        np.add.at(g, fz, -fadv[:, None] * pi2[fz])
        theta += lr * g / batch
        e = k + 1
        pi2n = np.exp(theta - theta.max(1, keepdims=True))
        pi2n /= pi2n.sum(1, keepdims=True)
        V += src.kernel_dist(src.induced_kernel(pi2n), src.induced_kernel(prev_step))
        prev_step = pi2n

        pop_mass, pop_success = src.exact_landmark_coverage(pi1, pi2n, H, symbol=1)
        log["population_upd"].append(e)
        log["population_mass_t1"].append(float(pop_mass))
        log["population_success"].append(float(pop_success))
        if pop_surv is None and pop_mass < 1.0 - delta:
            pop_surv = e

        if e % meas == 0:
            tr, sc = src.rollout(pi1, pi2n, nprobe, H, r_eval)
            m, _ = src.landmark_mass(tr, sc)
            inL = 1 in src.delta_landmarks(m, delta)
            log["upd"].append(e); log["V"].append(V); log["mass_t1"].append(float(m[1]))
            log["population_mass_at_eval"].append(float(pop_mass))
            log["in_L"].append(int(inL))
            if grid_pop_surv is None and pop_mass < 1.0 - delta:
                grid_pop_surv = e
            if surv is None and not inL:
                surv = e
    log["survival"] = surv if surv is not None else n_upd
    log["population_survival"] = pop_surv if pop_surv is not None else n_upd + 1
    log["grid_population_survival"] = (
        grid_pop_surv if grid_pop_surv is not None else n_upd + 1
    )
    return log


def e4():
    etas = [0.25, 0.5, 1.0, 2.0, 4.0]
    seeds = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    S = np.zeros((len(etas), len(seeds)))
    S_pop = np.zeros((len(etas), len(seeds)))
    S_grid = np.zeros((len(etas), len(seeds)))
    Vt = np.zeros((len(etas), len(seeds)))
    coverage_sqerr = []
    curves = {}
    for i, et in enumerate(etas):
        for j, s in enumerate(seeds):
            lg = run_pg(et, s)
            S[i, j] = lg["survival"]
            S_pop[i, j] = lg["population_survival"]
            S_grid[i, j] = lg["grid_population_survival"]
            Vt[i, j] = lg["V"][-1]
            coverage_sqerr.extend(
                (np.array(lg["mass_t1"]) - np.array(lg["population_mass_at_eval"])) ** 2
            )
            if j == 0:
                curves[et] = {
                    "eval_upd": np.array(lg["upd"]),
                    "estimated": np.array(lg["mass_t1"]),
                    "population_upd": np.array(lg["population_upd"]),
                    "population": np.array(lg["population_mass_t1"]),
                    "population_success": np.array(lg["population_success"]),
                }
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.12, 0.82, len(etas)))  # avoid teal (reserved for the core)
    for et, color in zip(etas, colors):
        curve = curves[et]
        keep_pop = curve["population_upd"] <= 250
        keep_eval = curve["eval_upd"] <= 250
        pop_upd = curve["population_upd"][keep_pop]
        pop_cov = curve["population"][keep_pop]
        ax[0].plot(pop_upd, pop_cov, lw=1.2, color=color)
        eval_upd = curve["eval_upd"][keep_eval]
        est = curve["estimated"][keep_eval]
        pop_succ = curve["population_success"][eval_upd]
        n_success = np.maximum(1.0, 5000.0 * pop_succ)
        half = 1.96 * np.sqrt(np.maximum(est * (1.0 - est), 0.0) / n_success)
        ax[0].errorbar(eval_upd, est, yerr=half, ls="none", marker="o", ms=2.0,
                       mfc="white", mec=color, ecolor=color, elinewidth=0.65,
                       capsize=1.2, alpha=0.78)
        label_idx = int(np.argmin(np.abs(pop_cov - 0.72)))
        label_dy = 8 if et == 4.0 else (-8 if et == 2.0 else 0)
        ax[0].annotate(rf"$\eta={et:g}$", xy=(pop_upd[label_idx], pop_cov[label_idx]),
                       xytext=(3, label_dy), textcoords="offset points", va="center",
                       fontsize=6.1, color=color)
    ax[0].axhline(0.9, ls=":", color="k", lw=0.8)
    ax[0].set_xlabel("peer update $e$"); ax[0].set_ylabel("support mass of $t_1$")
    ax[0].set_xlim(0, 250)
    ax[0].set_ylim(0.43, 1.01)
    ax[0].set_title("(a) Coverage erodes predictably", loc="left", fontweight="bold")
    ax[0].text(3, 0.445, "curves: exact population; points: 5,000 rollouts, 95% CI",
               fontsize=6.5, color=C["grey"])
    grid(ax[0])

    inv = 1.0 / np.array(etas)
    pop_mean, pop_lo, pop_hi = mean_ci95(S_pop, axis=1)
    est_mean, est_lo, est_hi = mean_ci95(S, axis=1)
    ax[1].errorbar(inv, pop_mean, yerr=[pop_mean - pop_lo, pop_hi - pop_mean], marker="o",
                   color=C["d1"], label="population exit")
    ax[1].errorbar(inv, est_mean, yerr=[est_mean - est_lo, est_hi - est_mean], marker="s",
                   color=C["d2"], label="evaluated exit")
    co_pop = np.polyfit(inv, S_pop.mean(1), 1)
    xs = np.linspace(0, inv.max() * 1.05, 20)
    ax[1].plot(xs, np.polyval(co_pop, xs), ls=":", color=C["grey"])
    rr_pop = np.corrcoef(inv, S_pop.mean(1))[0, 1] ** 2
    rr_est = np.corrcoef(inv, S.mean(1))[0, 1] ** 2
    ax[1].set_xlabel(r"$\frac{1}{\eta}$"); ax[1].set_ylabel("first-exit update")
    ax[1].set_title(rf"(b) Exit $\propto \frac{{1}}{{\eta}}$ ($R^2={rr_pop:.3f}$)",
                    loc="left", fontweight="bold")
    ax[1].legend(fontsize=6.4, title=r"mean $\pm$ 95% CI", title_fontsize=6.4)
    grid(ax[1])

    gap = S - S_pop
    grid_delay = S_grid - S_pop
    gap_mean, gap_lo, gap_hi = mean_ci95(gap, axis=1)
    delay_mean, delay_lo, delay_hi = mean_ci95(grid_delay, axis=1)
    ax[2].errorbar(etas, gap_mean, yerr=[gap_mean-gap_lo, gap_hi-gap_mean], marker="o",
                   color=C["d3"], label="rollout = grid delay")
    ax[2].axhline(0, ls=":", color="k", lw=0.8)
    ax[2].set_xlabel(r"peer step size $\eta$")
    ax[2].set_ylabel(r"$\widehat E_{\rm exit}-E_{\rm exit}$")
    ax[2].set_title("(c) Estimation adds no crossing delay", loc="left", fontweight="bold")
    ax[2].legend(fontsize=6.4, title=r"mean $\pm$ 95% CI, 10 seeds",
                 title_fontsize=6.2)
    grid(ax[2])
    fig.savefig(OUT + "fig_pg.pdf"); plt.close(fig)

    res["e4"] = {
        "etas": etas,
        "estimated_survival_mean": S.mean(1).tolist(),
        "estimated_survival_std": S.std(1).tolist(),
        "population_survival_mean": S_pop.mean(1).tolist(),
        "population_survival_std": S_pop.std(1).tolist(),
        "exit_error_mean": gap.mean(1).tolist(),
        "exit_error_std": gap.std(1).tolist(),
        "exit_mae_all": float(np.abs(gap).mean()),
        "evaluation_grid_delay_mean": grid_delay.mean(1).tolist(),
        "finite_minus_grid_mean": (S - S_grid).mean(1).tolist(),
        "finite_matches_grid_fraction": float(np.mean(S == S_grid)),
        "coverage_rmse_at_evaluations": float(np.sqrt(np.mean(coverage_sqerr))),
        "V_mean": Vt.mean(1).tolist(),
        "R2_population_survival_vs_inv_eta": float(rr_pop),
        "R2_estimated_survival_vs_inv_eta": float(rr_est),
        "product_eta_times_population_survival":
            [float(e * s) for e, s in zip(etas, S_pop.mean(1))],
        "product_eta_times_estimated_survival":
            [float(e * s) for e, s in zip(etas, S.mean(1))],
    }
    print("E4", json.dumps(res["e4"], indent=2))


if __name__ == "__main__":
    e3()
    e4()
    with open(OUT + "results2.json", "w") as f:
        json.dump(res, f, indent=2)
