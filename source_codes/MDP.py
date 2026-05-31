import numpy as np

class LinearMixtureTabularMDP:
    """Tabular MDP as linear mixture with one-hot basis."""
    def __init__(self, S, A, H, P, r, s1=0):
        self.S, self.A, self.H = S, A, H
        self.P = P          # (S,A,S)
        self.r = r          # (S,A)
        self.s1 = s1

    def step(self, s, a, rng: np.random.Generator):
        sp = rng.choice(self.S, p=self.P[s, a])
        return int(sp), float(self.r[s, a])

    def optimal_value(self):
        V = np.zeros((self.H + 2, self.S))
        pi = np.zeros((self.H + 1, self.S), dtype=int)
        for h in range(self.H, 0, -1):
            Q = self.r + self.P @ V[h + 1]
            pi[h] = np.argmax(Q, axis=1)
            V[h] = Q[np.arange(self.S), pi[h]]
        return V, pi

    def policy_value(self, pi: np.ndarray):
        V = np.zeros((self.H + 2, self.S))
        for h in range(self.H, 0, -1):
            a = pi[h]
            V[h] = self.r[np.arange(self.S), a] + (self.P[np.arange(self.S), a] @ V[h + 1])
        return V


def sample_offline_online_pair(if_shuffle, S, A, H, Delta, seed, s1=0, warn=True):
    """
    Sample online (P,r) and offline (P_off,r) with per-(s,a) L1 constraint check:
        ||P_off[s,a]-P[s,a]||_1 <= Delta
    Construction: add scaled Gaussian noise (pre-proj L1 = Delta/2), project to simplex,
    then double-check; if violated, revert that (s,a) block to P[s,a].
    """
    rng = np.random.default_rng(seed)

    P = rng.dirichlet(np.ones(S), size=(S, A))
    r = rng.uniform(0.0, 1.0, size=(S, A))

    P_off = np.zeros_like(P)
    degraded = 0
    max_excess = 0.0


    if not if_shuffle:
        for s in range(S):
            for a in range(A):
                z = rng.normal(size=S)
                z_l1 = np.linalg.norm(z, ord=1)
                if z_l1 < 1e-12:
                    z = rng.normal(size=S)
                    z_l1 = np.linalg.norm(z, ord=1)
    
                # scale so that ||0.5 * scale * z||_1 = Delta/2  => scale = Delta / ||z||_1
                scale = Delta / z_l1
                cand = simplex_projection(P[s, a] + 0.5 * scale * z)
    
                l1_dist = float(np.linalg.norm(cand - P[s, a], ord=1))
                if l1_dist > Delta + 1e-10:
                    degraded += 1
                    max_excess = max(max_excess, l1_dist - Delta)
                    P_off[s, a] = P[s, a].copy()
                else:
                    P_off[s, a] = cand

    else:
        for s in range(S):
            for a in range(A):
                p = P[s, a].copy()
        
                # ---- adversarial target q: "flip probability mass" by permutation ----
                # Strategy: map largest probs to positions of smallest probs (maximizes rearrangement)
                idx_asc = np.argsort(p)          # smallest -> largest indices
                idx_desc = idx_asc[::-1]         # largest -> smallest indices
        
                # q will place p's largest mass onto idx_asc positions
                # Equivalent: q[idx_asc] = p[idx_desc]
                q = np.empty_like(p)
                q[idx_asc] = p[idx_desc]
        
                # L1 distance if we fully permute
                l1_full = float(np.linalg.norm(q - p, ord=1))
        
                if l1_full < 1e-12:
                    # Degenerate case (all entries equal): any permutation is identical.
                    # Fall back to keeping p unchanged.
                    P_off[s, a] = p
                    degraded += 1
                    continue
        
                # ---- move toward q but cap L1 distance by Delta ----
                alpha = min(1.0, Delta / l1_full)
                cand = (1.0 - alpha) * p + alpha * q  # stays on simplex, nonnegative, sums to 1
        
                # numerical safety (optional)
                cand = np.clip(cand, 0.0, 1.0)
                cand = cand / max(1e-12, cand.sum())
        
                # double-check
                l1_dist = float(np.linalg.norm(cand - p, ord=1))
                if l1_dist > Delta + 1e-10:
                    # This should basically never happen; keep the old degrade logic anyway.
                    degraded += 1
                    max_excess = max(max_excess, l1_dist - Delta)
                    P_off[s, a] = p
                else:
                    P_off[s, a] = cand

    if warn:
        frac = degraded / (S * A)
        if degraded > 0:
            print(f"[warning] reverted {degraded}/{S*A} (s,a) blocks "
                  f"({100.0*frac:.2f}%) because ||P_off-P||_1 > Delta. "
                  f"Max excess over Delta = {max_excess:.3e}")
        else:
            print(f"[info] reverted 0/{S*A} (s,a) blocks (0.00%).")

    mdp_on  = LinearMixtureTabularMDP(S, A, H, P, r, s1)
    mdp_off = LinearMixtureTabularMDP(S, A, H, P_off, r.copy(), s1)
    return mdp_off, mdp_on

def sample_online_mdp(if_shuffle, S, A, H, seed, s1=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S), size=(S, A))
    r = rng.uniform(0.0, 1.0, size=(S, A))
    mdp_on = LinearMixtureTabularMDP(S, A, H, P, r, s1)
    return mdp_on

def sample_offline_from_online(mdp_on: LinearMixtureTabularMDP, if_shuffle: int, Delta: float,
                              seed: int, warn: bool = True):

    rng = np.random.default_rng(seed)

    S, A, H = mdp_on.S, mdp_on.A, mdp_on.H
    P = mdp_on.P
    r = mdp_on.r

    P_off = np.zeros_like(P)
    degraded = 0
    max_excess = 0.0

    if not if_shuffle:
        for s in range(S):
            for a in range(A):
                z = rng.normal(size=S)
                z_l1 = np.linalg.norm(z, ord=1)
                if z_l1 < 1e-12:
                    z = rng.normal(size=S)
                    z_l1 = np.linalg.norm(z, ord=1)

                scale = Delta / z_l1
                cand = simplex_projection(P[s, a] + 0.5 * scale * z)

                l1_dist = float(np.linalg.norm(cand - P[s, a], ord=1))
                if l1_dist > Delta + 1e-10:
                    degraded += 1
                    max_excess = max(max_excess, l1_dist - Delta)
                    P_off[s, a] = P[s, a].copy()
                else:
                    P_off[s, a] = cand
    else:
        for s in range(S):
            for a in range(A):
                p = P[s, a].copy()

                idx_asc = np.argsort(p)
                idx_desc = idx_asc[::-1]

                q = np.empty_like(p)
                q[idx_asc] = p[idx_desc]

                l1_full = float(np.linalg.norm(q - p, ord=1))
                if l1_full < 1e-12:
                    P_off[s, a] = p
                    degraded += 1
                    continue

                alpha = min(1.0, Delta / l1_full)
                cand = (1.0 - alpha) * p + alpha * q

                cand = np.clip(cand, 0.0, 1.0)
                cand = cand / max(1e-12, cand.sum())

                l1_dist = float(np.linalg.norm(cand - p, ord=1))
                if l1_dist > Delta + 1e-10:
                    degraded += 1
                    max_excess = max(max_excess, l1_dist - Delta)
                    P_off[s, a] = p
                else:
                    P_off[s, a] = cand

    if warn:
        frac = degraded / (S * A)
        if degraded > 0:
            print(f"[warning] reverted {degraded}/{S*A} (s,a) blocks "
                  f"({100.0*frac:.2f}%) because ||P_off-P||_1 > Delta. "
                  f"Max excess over Delta = {max_excess:.3e}")
        else:
            print(f"[info] reverted 0/{S*A} (s,a) blocks (0.00%).")

    mdp_off = LinearMixtureTabularMDP(S, A, H, P_off, r.copy(), mdp_on.s1)
    return mdp_off


def sample_offline_from_online1(
    mdp_on: LinearMixtureTabularMDP,
    if_shuffle: int,
    Delta: float,
    seed: int,
    warn: bool = True,
):
    """
    Adversarially construct P_off from P_on under per-(s,a) L1 <= Delta,
    by pushing probability mass toward value-worst states.

    This REPLACES the old random / shuffle construction.
    """

    rng = np.random.default_rng(seed)

    S, A, H = mdp_on.S, mdp_on.A, mdp_on.H
    P = mdp_on.P
    r = mdp_on.r

    # -----------------------------
    # 1) compute a "badness score" v(s)
    #    use WORST policy value (can switch to V*)
    # -----------------------------
    V = np.zeros((H + 2, S))
    for h in range(H, 0, -1):
        Q = r + P @ V[h + 1]
        V[h] = np.min(Q, axis=1)   # worst-value DP

    # pick one horizon slice to align corruption (any h is fine)
    v = V[1]   # shape (S,)

    # -----------------------------
    # 2) construct P_off adversarially
    # -----------------------------
    P_off = np.zeros_like(P)
    degraded = 0
    max_excess = 0.0

    for s in range(S):
        for a in range(A):
            p = P[s, a].copy()
            q = p.copy()

            # L1 budget: moving ε mass costs 2epsilon in L1
            budget = 0.5 * Delta

            # donors: high v; receivers: low v
            donors = np.argsort(-v)
            recvs  = np.argsort(v)

            i = j = 0
            while budget > 1e-12 and i < S and j < S:
                d = donors[i]
                rcv = recvs[j]

                if d == rcv:
                    j += 1
                    continue

                take = q[d]
                give_cap = 1.0 - q[rcv]
                if take <= 1e-12:
                    i += 1
                    continue
                if give_cap <= 1e-12:
                    j += 1
                    continue

                move = min(budget, take, give_cap)
                q[d]   -= move
                q[rcv] += move
                budget -= move

                if q[d] <= 1e-12:
                    i += 1
                if q[rcv] >= 1.0 - 1e-12:
                    j += 1

            # numerical cleanup
            q = np.clip(q, 0.0, 1.0)
            q_sum = q.sum()
            if q_sum <= 1e-12:
                q = np.ones(S) / S
            else:
                q /= q_sum

            l1_dist = float(np.linalg.norm(q - p, ord=1))
            if l1_dist > Delta + 1e-10:
                degraded += 1
                max_excess = max(max_excess, l1_dist - Delta)
                P_off[s, a] = p
            else:
                P_off[s, a] = q

    if warn:
        frac = degraded / (S * A)
        if degraded > 0:
            print(
                f"[warning] reverted {degraded}/{S*A} (s,a) blocks "
                f"({100.0*frac:.2f}%) because ||P_off-P||_1 > Delta. "
                f"Max excess over Delta = {max_excess:.3e}"
            )
        else:
            print("[info] reverted 0/(S*A) blocks (0.00%).")

    return LinearMixtureTabularMDP(S, A, H, P_off, r.copy(), mdp_on.s1)



def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Project v onto probability simplex {x>=0, sum x = 1}."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0]
    if rho.size == 0:
        return np.ones_like(v) / len(v)
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 1e-12:
        return np.ones_like(v) / len(v)
    return w / s



def _redistribute_mass(base_probs, target_idx, mass_to_add):
    if mass_to_add <= 0:
        return base_probs
    w = base_probs[target_idx].copy()
    w_sum = w.sum()
    if w_sum > 1e-12:
        base_probs[target_idx] += mass_to_add * (w / w_sum)
    else:
        base_probs[target_idx] += mass_to_add / len(target_idx)
    return base_probs


def impose_two_region_coverage_structure(P, P_off, s0=0, split=None, eps=1e-12):
    """
    Hard impose:
      - S split into S1=[0..split-1], S2=[split..S-1]
      - for s in S1:
          a!=0 : transitions stay in S1
          a==0 : transitions go to S2
      - for s in S2:
          any a : transition deterministically to s0 in S1
    """
    S, A, _ = P.shape
    if split is None:
        split = S // 2
    split = int(split)
    assert 1 <= split <= S - 1
    S1 = np.arange(0, split, dtype=int)
    S2 = np.arange(split, S, dtype=int)
    assert s0 in S1

    def _structured_version(P_in):
        P_new = np.zeros_like(P_in)

        for s in S1:
            for a in range(A):
                p = P_in[s, a].copy()
                if a == 0:
                    mass_S1 = p[S1].sum()
                    p[S1] = 0.0
                    p = _redistribute_mass(p, S2, mass_S1)
                else:
                    mass_S2 = p[S2].sum()
                    p[S2] = 0.0
                    p = _redistribute_mass(p, S1, mass_S2)

                ssum = p.sum()
                if ssum <= eps:
                    if a == 0:
                        p[S2] = 1.0 / len(S2)
                    else:
                        p[S1] = 1.0 / len(S1)
                else:
                    p /= ssum
                P_new[s, a] = p

        for s in S2:
            for a in range(A):
                p = np.zeros(S)
                p[s0] = 1.0
                P_new[s, a] = p

        return P_new

    return _structured_version(P), _structured_version(P_off)