import numpy as np

class RandomBehaviorPolicy:
    def __init__(self, A):
        self.A = A

    def act(self, s, h, rng):
        return int(rng.integers(self.A))


class BiasedCoverageBehaviorPolicy:
    """
    When s in S1:
        choose a=0 with prob z, else choose uniformly from {1,...,A-1}
    When s in S2:
        choose uniformly from {0,...,A-1}
    """
    def __init__(self, A, S1_states, z):
        self.A = int(A)
        self.S1 = set(int(x) for x in S1_states)
        self.z = float(z)
        assert 0.0 <= self.z <= 1.0
        assert self.A >= 2

    def act(self, s, h, rng):
        s = int(s)
        if s in self.S1:
            if rng.random() < self.z:
                return 0
            return int(rng.integers(1, self.A))
        return int(rng.integers(self.A))

class TildeVDesigner:
    def __init__(self, S, A):
        self.S, self.A = S, A

    def design(self, coverage_obs, z_action0, states, actions, mode, rng):
        """
        return: Vtilde (M_off, H+2, S)

        Updated "optimal" when coverage_obs==0:
          - fix a state subset S1 (default: first half states)
          - for (m,h) where s in S1:
              with prob z_action0: do the original "round-robin" one-hot
              with prob 1-z_action0: keep all-zero vector
          - for s not in S1: keep original behavior (always place one-hot)
        """
        M, Hp1 = states.shape
        H = Hp1 - 1
        V = np.zeros((M, H + 2, self.S), dtype=float)

        if mode == "zero":
            return V

        if mode == "random":
            V[:, 1:H + 1, :] = rng.random((M, H, self.S))
            return V

        if mode == 'pessimistic':
            return V

        if mode != "optimal":
            raise ValueError(f"Unknown tildeV_mode={mode}")

        # --- updated optimal ---
        # Round-robin counters as before
        counts = np.zeros((self.S, self.A, self.S), dtype=int)

        # coverage-controlled state subset
        if coverage_obs == 0:
            S1 = set(range(self.S // 2))  # consistent with the two-region design
            p_keep = float(np.clip(z_action0, 0.0, 1.0))
        else:
            S1 = None
            p_keep = 1.0  # always keep

        for m in range(M):
            for h in range(1, H + 1):
                s = int(states[m, h - 1])
                a = int(actions[m, h - 1])

                # If we are in coverage-controlled mode and s in S1:
                # keep this probe with prob z_action0, else keep all zeros
                if S1 is not None and (s in S1):
                    if rng.random() > p_keep:
                        continue  # leave V[m,h,:] = 0

                sp = int(np.argmin(counts[s, a]))
                counts[s, a, sp] += 1
                V[m, h, sp] = 1.0

        return V