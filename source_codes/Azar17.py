import numpy as np

class UCBVI_CH_Tabular:
    """
    UCBVI (Azar17) with:
      - global counts N_k(s,a) (no h index)
      - bonus_1: b(s,a)=7 H L sqrt(1/N_k(s,a))
      - monotone (non-increasing) Q update as in Algorithm 2:
            Q_k,h(s,a) = min(Q_{k-1,h}(s,a), H, rhat + Phat V_{k,h+1} + bonus)
        and if N_k(s,a)=0 then Q_k,h(s,a)=H
    """
    def __init__(self, mdp_on, cfg):
        self.mdp = mdp_on
        self.cfg = cfg
        self.S, self.A, self.H = cfg.S, cfg.A, cfg.H

        # global counts over (s,a) and (s,a,s')
        self.N_sa  = np.zeros((self.S, self.A), dtype=int)
        self.R_sum = np.zeros((self.S, self.A), dtype=float)
        self.N_sas = np.zeros((self.S, self.A, self.S), dtype=int)

        # store previous optimistic Q for monotone min (indexed by h)
        # Initialize Q_prev to H everywhere (max optimism).
        self.Q_prev = np.full((self.H + 1, self.S, self.A), float(self.H), dtype=float)

    def _L(self):
        cfg = self.cfg
        T = max(1, cfg.K * cfg.H)
        return np.log(max(5.0 * cfg.S * cfg.A * T / cfg.delta_conf, 1000.0))

    def run(self, rng: np.random.Generator):
        cfg = self.cfg

        Vstar, _ = self.mdp.optimal_value()
        Vstar_s1 = float(Vstar[1, self.mdp.s1])
        regret = np.zeros(cfg.K, dtype=float)

        # working buffers
        V = np.zeros((cfg.H + 2, cfg.S), dtype=float)
        Q = np.zeros((cfg.H + 1, cfg.S, cfg.A), dtype=float)

        L = self._L()
        c = 7.0  # bonus_1 constant

        for k in range(1, cfg.K + 1):
            V.fill(0.0)
            Q.fill(0.0)

            if k == 90:
                ass = 1

            # planning (Algorithm 2 style)
            for h in range(cfg.H, 0, -1):
                Vnext = V[h + 1]

                for s in range(cfg.S):
                    for a in range(cfg.A):
                        n = int(self.N_sa[s, a])

                        if n <= 0:
                            # unseen => Q_k,h = H
                            q = float(self.H)
                        else:
                            rhat = self.R_sum[s, a] / n
                            Phat = self.N_sas[s, a] / n

                            bonus = cfg.bonus_scale * c * cfg.H * L * np.sqrt(1.0 / n)
                            q_ucb = float(rhat + float(Phat @ Vnext) + bonus)

                            # monotone non-increasing + hard cap at H
                            q = min(float(self.Q_prev[h, s, a]), float(self.H), q_ucb)

                        if cfg.clip_value:
                            # the existing per-step horizon clip
                            q = float(np.clip(q, 0.0, cfg.H - h + 1))

                        Q[h, s, a] = q

                V[h] = np.max(Q[h], axis=1)

            # after planning, update Q_prev for next episode
            self.Q_prev[...] = Q

            # greedy policy (tie-break random)
            pi = np.zeros((cfg.H + 1, cfg.S), dtype=int)
            for hh in range(1, cfg.H + 1):
                for s in range(cfg.S):
                    row = Q[hh, s]
                    m = row.max()
                    candidates = np.flatnonzero(np.isclose(row, m, rtol=1e-12, atol=1e-12))
                    pi[hh, s] = int(rng.choice(candidates))

            # regret w.r.t. V*
            Vpi = self.mdp.policy_value(pi)
            regret[k - 1] = Vstar_s1 - float(Vpi[1, self.mdp.s1])

            # rollout + global update
            s = int(self.mdp.s1)
            for hh in range(1, cfg.H + 1):
                a = int(pi[hh, s])
                sp, _ = self.mdp.step(s, a, rng)

                r = float(self.mdp.r[s, a])
                self.N_sa[s, a] += 1
                self.R_sum[s, a] += r
                self.N_sas[s, a, int(sp)] += 1

                s = int(sp)

        return regret