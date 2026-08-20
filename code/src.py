"""
Switch-Rock Corridor (SRC): a minimal two-player decentralised Markov game used to
test the delta-core stability theory.

Focal states f: 0=s0, 1=t1, 2=t2, 3=b1, 4=b2, 5=g
Joint state z = (f, rock) with rock in {0,1}.  |Z| = 12.
Focal actions A1 = {0:up, 1:down, 2:fwd, 3:wait}
Peer actions  A2 = {0:hold, 1:clear, 2:idle}

Dynamics (one step, peer action a2 drawn from pi2(.|z)):
  hold  = (a2 == 0)
  clear = (a2 == 1)
  rock' = rock OR clear
  f=0 : up->t1, down->b1, else stay
  f=1 : fwd AND hold  -> t2, else stay      (top gate needs peer to hold switch)
  f=2 : fwd -> g
  f=3 : fwd AND rock==1 -> b2, else stay    (bottom gate needs rock cleared)
  f=4 : fwd -> g
  f=5 : absorbing goal

Abstraction phi = projection onto focal position f. Sigma = {s0,t1,t2,b1,b2,g}.

Key property (the discontinuity construction):
  alpha = 0  (peer never clears)  -> bottom route impossible -> t1,t2 are landmarks
  alpha > 0  (peer clears w.p. alpha) -> bottom route has positive probability
             -> support core (delta=0) loses t1,t2 immediately, for ANY alpha>0.
"""

import numpy as np

NF = 6
SYM = ["s0", "t1", "t2", "b1", "b2", "g"]
NZ = NF * 2          # (f, rock)
NA1 = 4
NA2 = 3
GOAL_F = 5


def zidx(f, rock):
    return f * 2 + rock


# ----------------------------------------------------------------------------- kernels
def joint_kernel():
    """P(z' | z, a1, a2) as a dense array [NZ, NA1, NA2, NZ]."""
    P = np.zeros((NZ, NA1, NA2, NZ))
    for f in range(NF):
        for rock in range(2):
            z = zidx(f, rock)
            for a1 in range(NA1):
                for a2 in range(NA2):
                    hold = (a2 == 0)
                    clear = (a2 == 1)
                    nrock = 1 if (rock == 1 or clear) else 0
                    if f == 0:
                        nf = 1 if a1 == 0 else (3 if a1 == 1 else 0)
                    elif f == 1:
                        # top gate needs peer to hold AND requires the rock intact;
                        # clearing the rock (for the bottom route) disables the top gate,
                        # so the two routes are mutually exclusive conventions.
                        nf = 2 if (a1 == 2 and hold and rock == 0) else 1
                    elif f == 2:
                        nf = 5 if a1 == 2 else 2
                    elif f == 3:
                        nf = 4 if (a1 == 2 and rock == 1) else 3
                    elif f == 4:
                        nf = 5 if a1 == 2 else 4
                    else:
                        nf, nrock = 5, rock
                    P[z, a1, a2, zidx(nf, nrock)] = 1.0
    return P


JP = joint_kernel()


def induced_kernel(pi2):
    """P_e(z'|z,a1) = sum_a2 P(z'|z,a1,a2) pi2(a2|z).  pi2: [NZ, NA2]."""
    return np.einsum("zijk,zj->zik", JP, pi2)


def pi2_alpha(alpha):
    """Peer family: clear w.p. alpha, hold w.p. 1-alpha, at every state."""
    pi2 = np.zeros((NZ, NA2))
    pi2[:, 0] = 1.0 - alpha
    pi2[:, 1] = alpha
    return pi2


def kernel_dist(P1, P2):
    """||P1 - P2||_{1,inf} = sup_{z,a1} sum_{z'} |P1 - P2|."""
    return np.abs(P1 - P2).sum(axis=-1).max()


def policy_dist(p1, p2):
    """sup_z || p1(.|z) - p2(.|z) ||_1."""
    return np.abs(p1 - p2).sum(axis=-1).max()


# ----------------------------------------------------------------------------- probe policy
def probe_policy(p_fwd=0.9):
    """Fixed measurement policy mu. Splits at s0, otherwise advances.

    The delta-core is a property of (M, phi, mu); mu must be declared.  We use a
    fixed probe so that measured drift is attributable to the peer, not to the
    focal agent's own learning."""
    pi1 = np.zeros((NZ, NA1))
    for f in range(NF):
        for rock in range(2):
            z = zidx(f, rock)
            if f == 0:
                pi1[z, 0] = 0.5
                pi1[z, 1] = 0.5
            elif f == GOAL_F:
                pi1[z, 3] = 1.0
            else:
                pi1[z, 2] = p_fwd
                pi1[z, 3] = 1.0 - p_fwd
    return pi1


# ----------------------------------------------------------------------------- rollouts
def rollout(pi1, pi2, n, H, rng):
    """Vectorised rollouts with label horizon H and at most H-1 transitions.

    Returns (traces[n,H] of focal symbols, success[n]); trace entry -1 marks an
    unused label slot or an already terminated episode."""
    f = np.zeros(n, dtype=np.int64)
    rock = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    traces = np.full((n, H), -1, dtype=np.int64)

    for t in range(max(H - 1, 0)):
        traces[:, t] = np.where(done, -1, f)
        z = f * 2 + rock
        u = rng.random(n)
        a1 = (u[:, None] > np.cumsum(pi1[z], axis=1)).sum(axis=1)
        u2 = rng.random(n)
        a2 = (u2[:, None] > np.cumsum(pi2[z], axis=1)).sum(axis=1)

        hold = (a2 == 0)
        clear = (a2 == 1)
        nf = f.copy()

        m = (f == 0)
        nf[m & (a1 == 0)] = 1
        nf[m & (a1 == 1)] = 3
        m = (f == 1)
        nf[m & (a1 == 2) & hold & (rock == 0)] = 2
        m = (f == 2)
        nf[m & (a1 == 2)] = 5
        m = (f == 3)
        nf[m & (a1 == 2) & (rock == 1)] = 4
        m = (f == 4)
        nf[m & (a1 == 2)] = 5

        nrock = np.maximum(rock, clear.astype(np.int64))
        f = np.where(done, f, nf)
        rock = np.where(done, rock, nrock)
        done = done | (f == GOAL_F)

    success = done
    # write the terminal goal symbol for successful traces that reached g
    return traces, success


# ----------------------------------------------------------------------------- exact coverage
def exact_landmark_coverage(pi1, pi2, H, symbol):
    """Exact population coverage of one focal-position symbol under fixed policies.

    The forward recursion augments the SRC state with a hit bit recording whether
    ``symbol`` has appeared among the decision labels.  It therefore computes the
    same success-conditional quantity as ``landmark_mass`` without rollout noise.
    """
    if not 0 <= symbol < NF:
        raise ValueError("symbol must be a valid focal-position index")

    P = induced_kernel(pi2)
    K = np.einsum("za,zak->zk", pi1, P)
    dist = np.zeros((NZ, 2), dtype=float)
    dist[zidx(0, 0), 0] = 1.0

    for _ in range(max(H - 1, 0)):
        marked = dist.copy()
        if symbol != GOAL_F:
            states = np.arange(NZ) // 2 == symbol
            marked[states, 1] += marked[states, 0]
            marked[states, 0] = 0.0
        dist = np.einsum("zh,zk->kh", marked, K)

    goal_states = np.arange(NZ) // 2 == GOAL_F
    p_succ = float(dist[goal_states].sum())
    if p_succ <= 0:
        return np.nan, 0.0
    if symbol == GOAL_F:
        return 1.0, p_succ
    hit_and_succeed = float(dist[goal_states, 1].sum())
    return hit_and_succeed / p_succ, p_succ


# ----------------------------------------------------------------------------- core estimation
def landmark_mass(traces, success):
    """Empirical mu-mass of each length-1 subsequence among SUCCESSFUL traces.

    Returns array [NF] of mu({tau : sigma in phi(tau)} | success)."""
    tr = traces[success]
    if len(tr) == 0:
        return np.full(NF, np.nan), 0
    mass = np.zeros(NF)
    for s in range(NF):
        mass[s] = (tr == s).any(axis=1).mean()
    # every successful trace terminates at g by construction
    mass[GOAL_F] = 1.0
    return mass, len(tr)


def delta_landmarks(mass, delta):
    """L_delta = {sigma : mass(sigma) >= 1 - delta}."""
    return set(int(s) for s in range(NF) if mass[s] >= 1.0 - delta - 1e-12)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pair_mass(traces, success, sig_a, sig_b):
    """Mass of the length-2 subsequence (sig_a, sig_b) among successes."""
    tr = traces[success]
    if len(tr) == 0:
        return np.nan
    hit = np.zeros(len(tr), dtype=bool)
    for i in range(len(tr)):
        row = tr[i]
        ia = np.where(row == sig_a)[0]
        if len(ia) == 0:
            continue
        ib = np.where(row == sig_b)[0]
        if len(ib) == 0:
            continue
        hit[i] = ib[-1] > ia[0]
    return hit.mean()
