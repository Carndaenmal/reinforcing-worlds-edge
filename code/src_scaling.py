"""
Generalized Switch-Rock Corridor (SRC-L-N).

A length-L corridor task embedded in an (N+1)-agent decentralised Markov game:
one focal agent + N peers. Generalises the 12-state/2-agent SRC used in the main
experiments so that we can sweep environment size L and agent count N and read off
how the *theory quantities* scale (stability-bound tightness, variation budget /
core survival, and core-computation cost).

DESIGN so that theory stays computable at every (L, N):
  * Focal position f in {s0, t_1..t_L (top), b_1..b_L (bottom), g}: 2L+2 positions.
  * A binary rock flag rho.  Focal joint state z=(f,rho): |Z| = 2*(2L+2).
  * The N peers act only through a single scalar sufficient statistic: the number
    m in {0..N} that choose "clear" this step (rock clears iff m>=1; the top gate
    needs the switch held, i.e. m==0 AND rho==0).  This keeps the induced kernel
    exact and of size independent of N's representation blow-up: the focal-induced
    kernel depends on the peers only through Pr[m>=1] = 1-(1-alpha)^N =: q(alpha,N).
  * Mutually exclusive routes: clearing the rock (m>=1) permanently disables the
    top gate (rho monotone up), exactly as in the base SRC.

Consequences used by the experiments:
  * Induced focal kernel P_e depends on peers only through q in [0,1].
  * ||P_{q} - P_{q'}||_{1,inf} = 2|q-q'|  (top vs bottom mass swap).
  * With N i.i.d. peers each clearing w.p. alpha, q = 1-(1-alpha)^N, so a fixed
    per-peer step eta produces an N-amplified world drift dq/dalpha = N(1-alpha)^{N-1}.
  * Top route is a chain s0->t_1->...->t_L->g of length L+1; its interior
    positions t_1..t_L are phi-bottlenecks at q=0, giving a core of size Theta(L).
"""

import numpy as np


class SRC:
    def __init__(self, L=2, N=1):
        """L = corridor length (>=1). N = number of peers (>=1)."""
        self.L = L
        self.N = N
        # positions: 0=s0 ; 1..L = t_1..t_L ; L+1..2L = b_1..b_L ; 2L+1 = g
        self.S0 = 0
        self.TOP = list(range(1, L + 1))
        self.BOT = list(range(L + 1, 2 * L + 1))
        self.G = 2 * L + 1
        self.NF = 2 * L + 2
        self.NZ = self.NF * 2
        self.SYM = (["s0"] + [f"t{i}" for i in range(1, L + 1)]
                    + [f"b{i}" for i in range(1, L + 1)] + ["g"])

    def zidx(self, f, rock):
        return f * 2 + rock

    # ---- focal-induced kernel as a function of q = Pr[some peer clears] ----
    def induced_kernel(self, q):
        """P(z'|z,a1) for focal actions a1 in {0:up,1:down,2:fwd,3:wait}.
        Peers enter only through q=Pr[m>=1]. Returns [NZ, 4, NZ]."""
        NZ, L = self.NZ, self.L
        P = np.zeros((NZ, 4, NZ))
        for f in range(self.NF):
            for rock in range(2):
                z = self.zidx(f, rock)
                for a1 in range(4):
                    # distribution over (next rock): clear happens w.p. q
                    for cleared, pc in ((1, q), (0, 1.0 - q)):
                        nrock = 1 if (rock == 1 or cleared == 1) else 0
                        nf = self._focal_next(f, a1, rock, cleared)
                        P[z, a1, self.zidx(nf, nrock)] += pc
        return P

    def _focal_next(self, f, a1, rock, cleared):
        L = self.L
        if f == self.S0:
            if a1 == 0:   # up
                return self.TOP[0]
            if a1 == 1:   # down
                return self.BOT[0]
            return f
        if f in self.TOP:
            i = self.TOP.index(f)
            # advance only if fwd, switch held (no peer clears this step) and rock intact
            if a1 == 2 and cleared == 0 and rock == 0:
                return self.TOP[i + 1] if i + 1 < L else self.G
            return f
        if f in self.BOT:
            i = self.BOT.index(f)
            # advance only if fwd and rock already cleared
            if a1 == 2 and rock == 1:
                return self.BOT[i + 1] if i + 1 < L else self.G
            return f
        return self.G  # absorbing

    # ---- probe policy ----
    def probe_policy(self, p_fwd=0.9):
        pi1 = np.zeros((self.NZ, 4))
        for f in range(self.NF):
            for rock in range(2):
                z = self.zidx(f, rock)
                if f == self.S0:
                    pi1[z, 0] = 0.5
                    pi1[z, 1] = 0.5
                elif f == self.G:
                    pi1[z, 3] = 1.0
                else:
                    pi1[z, 2] = p_fwd
                    pi1[z, 3] = 1.0 - p_fwd
        return pi1

    # ---- vectorised rollouts under induced q ----
    def rollout(self, pi1, q, n, H, rng):
        f = np.zeros(n, dtype=np.int64)
        rock = np.zeros(n, dtype=np.int64)
        done = np.zeros(n, dtype=bool)
        traces = np.full((n, H), -1, dtype=np.int64)
        for t in range(max(H - 1, 0)):
            traces[:, t] = np.where(done, -1, f)
            z = f * 2 + rock
            u = rng.random(n)
            a1 = (u[:, None] > np.cumsum(pi1[z], axis=1)).sum(axis=1)
            cleared = (rng.random(n) < q).astype(np.int64)
            nf = f.copy()
            m = (f == self.S0)
            nf[m & (a1 == 0)] = self.TOP[0]
            nf[m & (a1 == 1)] = self.BOT[0]
            for i, ff in enumerate(self.TOP):
                nxt = self.TOP[i + 1] if i + 1 < self.L else self.G
                m = (f == ff) & (a1 == 2) & (cleared == 0) & (rock == 0)
                nf[m] = nxt
            for i, ff in enumerate(self.BOT):
                nxt = self.BOT[i + 1] if i + 1 < self.L else self.G
                m = (f == ff) & (a1 == 2) & (rock == 1)
                nf[m] = nxt
            nrock = np.maximum(rock, cleared)
            f = np.where(done, f, nf)
            rock = np.where(done, rock, nrock)
            done = done | (f == self.G)
        return traces, done

    # ---- coverage / delta-core over length-1 abstract symbols ----
    def landmark_mass(self, traces, success):
        tr = traces[success]
        if len(tr) == 0:
            return np.full(self.NF, np.nan), 0
        mass = np.zeros(self.NF)
        for s in range(self.NF):
            mass[s] = (tr == s).any(axis=1).mean()
        mass[self.G] = 1.0
        return mass, len(tr)

    def delta_core(self, mass, delta):
        return set(int(s) for s in range(self.NF)
                   if mass[s] >= 1.0 - delta - 1e-12)


# ---- peer aggregation ----
def q_of_alpha(alpha, N):
    """Pr at least one of N i.i.d. peers clears, each w.p. alpha."""
    return 1.0 - (1.0 - alpha) ** N


def kernel_dist_q(env, q1, q2):
    P1 = env.induced_kernel(q1)
    P2 = env.induced_kernel(q2)
    return np.abs(P1 - P2).sum(axis=-1).max()


# ---- exact multi-sequence LCS core cost instrumentation ----
def lcs_core_bruteforce(sequences, counter=None):
    """Exact longest common subsequence over k sequences by DP on the product
    index space. Returns (lcs_length, n_cells_filled). Used ONLY to measure the
    empirical O(H^k) cost that motivates abstraction; not used at scale."""
    if not sequences:
        return 0, 0
    dp = {}
    cells = 0

    def rec(idx):
        nonlocal cells
        if any(idx[i] == len(sequences[i]) for i in range(len(sequences))):
            return 0
        if idx in dp:
            return dp[idx]
        cells += 1
        syms = {sequences[i][idx[i]] for i in range(len(sequences))}
        best = 0
        if len(syms) == 1:
            nxt = tuple(idx[i] + 1 for i in range(len(sequences)))
            best = 1 + rec(nxt)
        else:
            for i in range(len(sequences)):
                nxt = list(idx)
                nxt[i] += 1
                best = max(best, rec(tuple(nxt)))
        dp[idx] = best
        return best

    val = rec(tuple(0 for _ in sequences))
    return val, cells
