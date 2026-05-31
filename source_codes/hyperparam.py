
from dataclasses import dataclass

import numpy as np

@dataclass
class SimConfig:

    plt_choice_list = ['A','B','C','D','E', 'legend']
    plt_choice = plt_choice_list[1]

    n_runs = 50
    if_running = 0

    seed: int = 3
    # tabular MDP size
    S: int = 5
    A: int = 10
    H: int = 3

    # offline / online sizes
    M_off: int = 100000      # offline trajectories
    K: int = 4000            # online episodes

    # environment shift (used by algorithm; offline env is still generated separately)
    Delta: float = 0.05
    if_shuffle: int = 1

    # algorithmic parameters
    lam: float = 1.0
    delta_conf: float = 0.0   # will be set to 1/K if <=0
    B: float = 2.0

    bonus_scale: float = 0.02

    # tilde-V mode: "optimal", "zero", "random"
    tildeV_mode: str = "optimal"

    # coverage setting:
    #   coverage_obs = 1 -> random offline policy
    #   coverage_obs = 0 -> coverage-controlled policy (two-region + biased action0)
    coverage_obs: int = 1
    z_action0: float = 0.0    # will be set to 1/sqrt(M_off) if <=0

    # merge mode: algorithm assumes offline==online by setting Delta=0 (shift term disappears)
    merge_mode: bool = False

    # optional clipping of optimistic value
    clip_value: bool = True

    online_algo: str = "ours"

    s1: int = 0

    def __post_init__(self):
        if self.delta_conf <= 0.0:
            self.delta_conf = 1.0 / max(1, self.K)
        if self.z_action0 <= 0.0:
            #self.z_action0 = 1.0 / np.sqrt(max(1, self.M_off))
            self.z_action0 = 0.3