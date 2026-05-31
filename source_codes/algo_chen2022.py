import numpy as np

class DataPooling_UCB_Theorem2_Tabular:
    """
    Tabular data pooling perturbed LSVI (UCB flavor): xi = 1.
    """
    def __init__(self, mdp_on, cfg, off_states, off_actions):
        self.mdp = mdp_on
        self.cfg = cfg
        self.off_states = off_states
        self.off_actions = off_actions

        self.S, self.A, self.H = cfg.S, cfg.A, cfg.H

        self.N_off = np.zeros((self.H + 1, self.S, self.A), dtype=int)
        self.R_off_sum = np.zeros((self.H + 1, self.S, self.A), dtype=float)
        self.P_off_cnt = np.zeros((self.H + 1, self.S, self.A, self.S), dtype=int)

        self.n_on = np.zeros((self.H + 1, self.S, self.A), dtype=int)
        self.R_on_sum = np.zeros((self.H + 1, self.S, self.A), dtype=float)
        self.P_on_cnt = np.zeros((self.H + 1, self.S, self.A, self.S), dtype=int)

        self._offline_phase_done = False

    def offline_phase0(self):
        if self._offline_phase_done:
            return
        cfg = self.cfg

        for m in range(cfg.M_off):
            for h in range(1, cfg.H + 1):
                s = int(self.off_states[m, h - 1])
                a = int(self.off_actions[m, h - 1])
                sp = int(self.off_states[m, h])

                r = float(self.mdp.r[s, a]) 
                self.N_off[h, s, a] += 1
                self.R_off_sum[h, s, a] += r
                self.P_off_cnt[h, s, a, sp] += 1

        self._offline_phase_done = True

    def offline_phase(self, rng):
        if self._offline_phase_done:
            return
    
        cfg = self.cfg
    
        # ------------------------------------------------------------
        # coverage-controlled state subset (same convention as before)
        # ------------------------------------------------------------
        if cfg.coverage_obs == 0:
            S1 = set(range(self.S // 2))  # consistent with the two-region design
            p_keep = float(np.clip(cfg.z_action0, 0.0, 1.0))
        else:
            S1 = None
            p_keep = 1.0
    
        for m in range(cfg.M_off):
            for h in range(1, cfg.H + 1):
                s = int(self.off_states[m, h - 1])
                a = int(self.off_actions[m, h - 1])
    
                # If coverage-controlled and s in S1:
                # keep this transition with prob z_action0, else skip
                if S1 is not None and (s in S1):
                    if rng.random() > p_keep:   # or use rng passed in / stored
                        continue
    
                sp = int(self.off_states[m, h])
    
                r = float(self.mdp.r[s, a])  # known reward table
                self.N_off[h, s, a] += 1
                self.R_off_sum[h, s, a] += r
                self.P_off_cnt[h, s, a, sp] += 1
    
        self._offline_phase_done = True

    def _log_term(self):
        cfg = self.cfg
        T = max(1, cfg.K)
        return np.log(max(2.0 * cfg.H * cfg.S * cfg.A * T / cfg.delta_conf, 1000.0))

    def _lambda_dp(self, n: int, N: int):
        cfg = self.cfg
        if cfg.Delta <= 0:
            return 1.0

        L = self._log_term()
        # threshold: log(2HSAT/delta)/(2 Δ^2)
        thr = L / (2.0 * (cfg.Delta ** 2))

        if n >= thr:
            return 1.0

        # lambda = ( n + N*n*Δ / sqrt( (N+n)*L/2 - Δ^2*N*n ) ) / (N+n)
        denom = float(N + n)
        if denom <= 0:
            return 0.0

        inside = (N + n) * L / 2.0 - (cfg.Delta ** 2) * N * n
        if inside <= 1e-12:
            # if inside <= 0, the formula is not real-valued; push lambda to 1
            return 1.0

        lam = (n + (N * n * cfg.Delta) / np.sqrt(inside)) / denom
        return float(np.clip(lam, 0.0, 1.0))

    def _epsilon_dp(self, lam: float, n: int, N: int):
        cfg = self.cfg
        L = self._log_term()

        nn = max(1, int(n))
        NN = max(1, int(N))

        term = (lam ** 2) / (2.0 * nn) + ((1.0 - lam) ** 2) / (2.0 * NN)
        rad = cfg.H * np.sqrt(max(0.0, L * term))
        shift = cfg.H * (1.0 - lam) * cfg.Delta
        return float(rad + shift)

    def run(self, rng: np.random.Generator):
        cfg = self.cfg
        self.offline_phase(rng)

        Vstar, _ = self.mdp.optimal_value()
        Vstar_s1 = float(Vstar[1, self.mdp.s1])
        regret = np.zeros(cfg.K, dtype=float)

        # DP value arrays
        Vtil = np.zeros((cfg.H + 2, cfg.S), dtype=float)
        Qtil = np.zeros((cfg.H + 1, cfg.S, cfg.A), dtype=float)

        for k in range(1, cfg.K + 1):
            if k == 90:
                sss = 1
            # -------- plan with current estimates --------
            Vtil.fill(0.0)
            Qtil.fill(0.0)

            for h in range(cfg.H, 0, -1):
                Vnext = Vtil[h + 1]

                for s in range(cfg.S):
                    for a in range(cfg.A):
                        n = int(self.n_on[h, s, a])
                        N = int(self.N_off[h, s, a])

                        lam = self._lambda_dp(n=n, N=N)

                        # online MLE (sample mean)
                        if n > 0:
                            rt = self.R_on_sum[h, s, a] / n
                            Pt = self.P_on_cnt[h, s, a] / n
                        else:
                            # no online data yet -> fallback to uniform (or zeros)
                            rt = 0.0
                            Pt = np.ones(cfg.S, dtype=float) / cfg.S

                        # offline MLE (sample mean)
                        if N > 0:
                            r0 = self.R_off_sum[h, s, a] / N
                            P0 = self.P_off_cnt[h, s, a] / N
                        else:
                            r0 = 0.0
                            P0 = np.ones(cfg.S, dtype=float) / cfg.S

                        rDP = lam * rt + (1.0 - lam) * r0
                        PDP = lam * Pt + (1.0 - lam) * P0

                        eps = self._epsilon_dp(lam=lam, n=n, N=N)
                        xi = 1.0  # UCB version
                        w = eps * xi

                        q = float(rDP + cfg.bonus_scale * w + float(PDP @ Vnext))

                        if cfg.clip_value:
                            q = float(np.clip(q, 0.0, cfg.H - h + 1))

                        Qtil[h, s, a] = q

                Vtil[h] = np.max(Qtil[h], axis=1)

            # greedy policy w.r.t. Qtil
            pi = np.zeros((cfg.H + 1, cfg.S), dtype=int)
            for hh in range(1, cfg.H + 1):
                for s in range(cfg.S):
                    row = Qtil[hh, s]
                    m = row.max()
                    candidates = np.flatnonzero(np.isclose(row, m, rtol=1e-12, atol=1e-12))
                    pi[hh, s] = int(rng.choice(candidates))

            Vpi = self.mdp.policy_value(pi)
            regret[k - 1] = Vstar_s1 - float(Vpi[1, self.mdp.s1])

            # -------- rollout + online update --------
            s = self.mdp.s1
            for hh in range(1, cfg.H + 1):
                a = int(pi[hh, s])
                sp, _ = self.mdp.step(s, a, rng)

                r = float(self.mdp.r[s, a])
                self.n_on[hh, s, a] += 1
                self.R_on_sum[hh, s, a] += r
                self.P_on_cnt[hh, s, a, int(sp)] += 1

                s = int(sp)

        return regret