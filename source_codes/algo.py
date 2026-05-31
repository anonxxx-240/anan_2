from MDP import *
from behavior_policy import *
from hyperparam import *

class OfflineOnlineUCRL_TabularBlocks:
    def __init__(self, mdp_on: LinearMixtureTabularMDP, cfg: SimConfig,
                 Vtilde, off_states, off_actions):
        self.mdp = mdp_on
        self.cfg = cfg
        self.Vtilde = Vtilde
        self.off_states = off_states
        self.off_actions = off_actions

        self.S, self.A, self.H = cfg.S, cfg.A, cfg.H
        self.d = self.S * self.A * self.S

        self.Mon = np.zeros((self.S, self.A, self.S, self.S), dtype=float)
        self.Mall = np.zeros((self.S, self.A, self.S, self.S), dtype=float)
        self.won = np.zeros((self.S, self.A, self.S), dtype=float)
        self.wall = np.zeros((self.S, self.A, self.S), dtype=float)

        for s in range(self.S):
            for a in range(self.A):
                self.Mon[s, a] = cfg.lam * np.eye(self.S)
                self.Mall[s, a] = cfg.lam * np.eye(self.S)

        self.Goff = np.zeros((self.S, self.A, self.S, self.S), dtype=float)
        self.merge_scale = 1

    def _logdet_sum_blocks(self, Mblocks):
        total = 0.0
        for s in range(self.S):
            for a in range(self.A):
                sign, ld = np.linalg.slogdet(Mblocks[s, a])
                if sign <= 0:
                    ld = np.log(max(np.linalg.det(Mblocks[s, a]), 1e-300))
                total += ld
        return float(total)

    def _lambda_max_Goff(self):
        lm = 0.0
        for s in range(self.S):
            for a in range(self.A):
                eigs = np.linalg.eigvalsh(self.Goff[s, a])
                lm = max(lm, float(eigs[-1]))
        return lm

    def _lambda_min_lamI_plus_Goff(self):
        lam = self.cfg.lam
        lmin = float("inf")
        for s in range(self.S):
            for a in range(self.A):
                eigs = np.linalg.eigvalsh(self.Goff[s, a] + lam * np.eye(self.S))
                lmin = min(lmin, float(eigs[0]))
        if not np.isfinite(lmin):
            lmin = lam
        return lmin

    def _compute_beta_gamma_strict(self):
        cfg = self.cfg

        log_det_lamI_half = 0.5 * self.d * np.log(cfg.lam)

        logdet_on = self._logdet_sum_blocks(self.Mon)
        logdet_all = self._logdet_sum_blocks(self.Mall)
        log_det_on_half = 0.5 * logdet_on
        log_det_all_half = 0.5 * logdet_all

        inside_beta = 2.0 * ((log_det_on_half - log_det_lamI_half) + np.log(1.0 / cfg.delta_conf))
        inside_beta = max(0.0, inside_beta)
        beta_k = cfg.H * np.sqrt(inside_beta) + np.sqrt(cfg.lam) * cfg.B

        lam_max_goff = self._lambda_max_Goff()
        lam_min_lamI_goff = self._lambda_min_lamI_plus_Goff()
        shift_term = cfg.Delta * (lam_max_goff / np.sqrt(max(1e-30, lam_min_lamI_goff)))

        inside_gamma = 2.0 * ((log_det_all_half - log_det_lamI_half) + np.log(cfg.K))
        inside_gamma = max(0.0, inside_gamma)
        gamma_k = cfg.H * np.sqrt(inside_gamma) + shift_term + np.sqrt(cfg.lam) * cfg.B

        return float(beta_k), float(gamma_k)

    def offline_phase_0(self):
        cfg = self.cfg
        for m in range(cfg.M_off):
            for h in range(1, cfg.H + 1):
                s = int(self.off_states[m, h - 1])
                a = int(self.off_actions[m, h - 1])
                sp = int(self.off_states[m, h])

                Vnext = self.Vtilde[m, h + 1]
                y = float(Vnext[sp])

                outer = np.outer(Vnext, Vnext)
                self.Goff[s, a] += outer
                self.Mall[s, a] += outer
                self.wall[s, a] += Vnext * y

    def offline_phase(self):
        """
        Offline phase.
        - If cfg.tildeV_mode != "pessimistic": keep the original behavior (use self.Vtilde probes).
        - If cfg.tildeV_mode == "pessimistic":
            For each offline trajectory m:
              (1) compute P_hat from current Mall/wall (built from previous trajectories)
              (2) run backward DP with NEGATIVE bonus to get Vp^m
              (3) update Goff/Mall/wall using this trajectory and Vp^m (online-flavor update)
        """
        cfg = self.cfg
        S, A, H = self.S, self.A, self.H
    
        # ------------------------------------------------------------
        # Case 1: keep the original offline_phase behavior
        # ------------------------------------------------------------
        if cfg.tildeV_mode != "pessimistic":
            for m in range(cfg.M_off):
                for h in range(1, H + 1):
                    s = int(self.off_states[m, h - 1])
                    a = int(self.off_actions[m, h - 1])
                    sp = int(self.off_states[m, h])
    
                    Vnext = self.Vtilde[m, h + 1]
                    y = float(Vnext[sp])
    
                    outer = np.outer(Vnext, Vnext)
                    self.Goff[s, a] += outer
                    self.Mall[s, a] += outer
                    self.wall[s, a] += Vnext * y
            return 0
    
        # ------------------------------------------------------------
        # Case 2: pessimistic design (online flavor, per-trajectory Vp^m)
        # ------------------------------------------------------------
        beta_off = float(getattr(cfg, "beta_off", 1.0))  # add cfg.beta_off if you want
        # optional: if you want to scale pessimism (like cfg.bonus_scale in online)
    
        # Workspace
        P_hat = np.zeros((S, A, S), dtype=float)
        Vp = np.zeros((H + 2, S), dtype=float)
        Qp = np.zeros((H + 1, S, A), dtype=float)
    
        for m in range(cfg.M_off):
            # 1) compute current P_hat from Mall/wall (built from trajectories < m)
            for s in range(S):
                for a in range(A):
                    P_hat[s, a] = np.linalg.solve(self.Mall[s, a], self.wall[s, a])
    
            # 2) backward DP with NEGATIVE bonus to get Vp^m
            Vp.fill(0.0)
            Qp.fill(0.0)
    
            for h in range(H, 0, -1):
                Vnext = Vp[h + 1]
                for s in range(S):
                    for a in range(A):
                        mu = float(P_hat[s, a] @ Vnext)
    
                        invV = np.linalg.solve(self.Mall[s, a], Vnext)
                        rad = float(np.sqrt(max(0.0, Vnext @ invV)))
    
                        pess = mu - beta_off * rad * cfg.bonus_scale
                        q = float(self.mdp.r[s, a] + pess)
    
                        if cfg.clip_value:
                            q = float(np.clip(q, 0.0, H - h + 1))
    
                        Qp[h, s, a] = q
    
                Vp[h] = np.max(Qp[h], axis=1)
    
            # 3) update using ONLY this trajectory m, with Vnext = Vp[h+1]
            for h in range(1, H + 1):
                s = int(self.off_states[m, h - 1])
                a = int(self.off_actions[m, h - 1])
                sp = int(self.off_states[m, h])
    
                Vnext = Vp[h + 1]
                y = float(Vnext[sp])
    
                outer = np.outer(Vnext, Vnext)
                self.Goff[s, a] += outer
                self.Mall[s, a] += outer
                self.wall[s, a] += Vnext * y

    def run(self, rng: np.random.Generator):
        cfg = self.cfg
        self.offline_phase()

        Vstar, _ = self.mdp.optimal_value()
        Vstar_s1 = float(Vstar[1, self.mdp.s1])
        regret = np.zeros(cfg.K)

        for k in range(1, cfg.K + 1):
            beta_k, gamma_k = self._compute_beta_gamma_strict()

            P_on_hat = np.zeros((self.S, self.A, self.S), dtype=float)
            P_all_hat = np.zeros((self.S, self.A, self.S), dtype=float)
            for s in range(self.S):
                for a in range(self.A):
                    P_on_hat[s, a] = np.linalg.solve(self.Mon[s, a], self.won[s, a])
                    P_all_hat[s, a] = np.linalg.solve(self.Mall[s, a], self.wall[s, a])



            Vhat = np.zeros((cfg.H + 2, cfg.S), dtype=float)
            Qhat = np.zeros((cfg.H + 1, cfg.S, cfg.A), dtype=float)

            for h in range(cfg.H, 0, -1):
                Vnext = Vhat[h + 1]
                for s in range(self.S):
                    for a in range(self.A):
                        mu_on = float(P_on_hat[s, a] @ Vnext)
                        mu_all = float(P_all_hat[s, a] @ Vnext)

                        inv_on_V = np.linalg.solve(self.Mon[s, a], Vnext)
                        inv_all_V = np.linalg.solve(self.Mall[s, a], Vnext)
                        rad_on = float(np.sqrt(max(0.0, Vnext @ inv_on_V)))
                        rad_all = float(np.sqrt(max(0.0, Vnext @ inv_all_V)))

                        ucb_on = mu_on + beta_k * rad_on * cfg.bonus_scale
                        ucb_all = mu_all + gamma_k * rad_all * cfg.bonus_scale


                        if cfg.tildeV_mode == 'zero':
                            bonus_val = ucb_on
                        elif not cfg.merge_mode:
                            bonus_val = min(ucb_on, ucb_all)
                        else:
                            bonus_val = ucb_all * self.merge_scale
                        q = float(self.mdp.r[s, a] +  bonus_val)

                        if cfg.clip_value:
                            q = float(np.clip(q, 0.0, cfg.H - h + 1))

                        Qhat[h, s, a] = q

                Vhat[h] = np.max(Qhat[h], axis=1)

            if k == 9000:
                sss = 1

            pi = np.zeros((cfg.H + 1, cfg.S), dtype=int)
            for hh in range(1, cfg.H + 1):
                for s in range(self.S):
                    row = Qhat[hh, s]
                    m = row.max()
                    candidates = np.flatnonzero(np.isclose(row, m, rtol=1e-12, atol=1e-12))
                    pi[hh, s] = int(rng.choice(candidates))

            Vpi = self.mdp.policy_value(pi)
            regret[k - 1] = Vstar_s1 - float(Vpi[1, self.mdp.s1])

            # online update
            s = self.mdp.s1
            for hh in range(1, cfg.H + 1):
                a = int(pi[hh, s])
                sp, _ = self.mdp.step(s, a, rng)

                Vnext = Vhat[hh + 1]
                y = float(Vnext[sp])
                outer = np.outer(Vnext, Vnext)

                self.Mon[s, a] += outer
                self.won[s, a] += Vnext * y
                self.Mall[s, a] += outer
                self.wall[s, a] += Vnext * y

                s = sp

        return regret


def collect_offline(mdp_off, behavior_policy, cfg: SimConfig, rng):
    states = np.zeros((cfg.M_off, cfg.H + 1), dtype=int)
    actions = np.zeros((cfg.M_off, cfg.H), dtype=int)

    for m in range(cfg.M_off):
        s = mdp_off.s1
        states[m, 0] = s
        for h in range(cfg.H):
            a = behavior_policy.act(s, h + 1, rng)
            actions[m, h] = a
            s, _ = mdp_off.step(s, a, rng)
            states[m, h + 1] = s
    return states, actions