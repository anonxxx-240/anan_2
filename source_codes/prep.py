# =========================
# runner + plotting (UPDATED, FIXED online MDP per run for sweeps)
# =========================

from algo import *
from plot import *
from algo_chen2022 import *
from Azar17 import *
from MDP import *
from behavior_policy import *
from hyperparam import *

import numpy as np
import matplotlib.pyplot as plt

import pickle
from pathlib import Path


def _get_algo_settings():
    settings = {
        "zero":    {"tildeV_mode": "zero",    "merge_mode": False, "online_algo": "ours",     "bonus_scale": 0.02},
        "optimal": {"tildeV_mode": "optimal", "merge_mode": False, "online_algo": "ours",     "bonus_scale": 0.02},
        "pessimistic":  {"tildeV_mode": "pessimistic",  "merge_mode": False, "online_algo": "ours", "bonus_scale": 0.02},
        "merge":   {"tildeV_mode": "optimal", "merge_mode": True,  "online_algo": "ours",     "bonus_scale": 0.015},

        "dp_ucb":  {"tildeV_mode": "zero",    "merge_mode": False, "online_algo": "dp_ucb_t2","bonus_scale": 0.05},
        "ucbvi":   {"tildeV_mode": "zero",    "merge_mode": False, "online_algo": "ucbvi",    "bonus_scale": 0.0002},
    }
    return settings

# ------------------------------------------------------
# 0) Shared helpers: summaries + plotting (UNCHANGED)
# ------------------------------------------------------
def summarize_curves(curves_runs: dict, q_lo=0.2, q_hi=0.8, cumulative: bool = True):
    """
    Input:
      curves_runs[name] = (n_runs, K) per-episode regret
    Output:
      summary[name] = dict with mean, lo, hi
    """
    summary = {}
    for name, regs in curves_runs.items():
        y = np.cumsum(regs, axis=1) if cumulative else regs
        mean = y.mean(axis=0)
        lo = np.quantile(y, q_lo, axis=0)
        hi = np.quantile(y, q_hi, axis=0)
        summary[name] = {"mean": mean, "lo": lo, "hi": hi}
    return summary


def summarize_final_at_K(curves_runs: dict, q_lo=0.2, q_hi=0.8, cumulative: bool = True):
    """
    Input:
      curves_runs[name] = (n_runs, K) per-episode regret
    Output:
      out[name] = dict with mean, lo, hi for scalar Y_K
        where Y_K = cumulative_regret(K) if cumulative else regret_at_K
    """
    out = {}
    for name, regs in curves_runs.items():
        if cumulative:
            y = np.cumsum(regs, axis=1)[:, -1]   # (n_runs,)
        else:
            y = regs[:, -1]                      # per-episode regret at K

        out[name] = {
            "mean": float(np.mean(y)),
            "lo":   float(np.quantile(y, q_lo)),
            "hi":   float(np.quantile(y, q_hi)),
        }
    return out


# ------------------------------------------------------
# 1) SINGLE RUN for a GIVEN (mdp_on, mdp_off) AND SHARED OFFLINE DATA
#    (This replaces the old run_setting/run_setting_given_mdps usage.)
# ------------------------------------------------------
def run_one_algo_given_everything(
    cfg_alg: SimConfig,
    mdp_on,
    off_states,
    off_actions,
    Vtilde,
    rng_algo: np.random.Generator,
):
    """
    Run exactly ONE algorithm using fixed:
      - mdp_on
      - offline dataset (off_states, off_actions)
      - Vtilde (precomputed from offline data)
    Returns: regret array shape (K,)
    """

    online_algo = getattr(cfg_alg, "online_algo", "ours")

    if online_algo == "dp_ucb_t2":
        algo = DataPooling_UCB_Theorem2_Tabular(
            mdp_on=mdp_on,
            cfg=cfg_alg,
            off_states=off_states,
            off_actions=off_actions,
        )
        return algo.run(rng_algo)

    if online_algo == "ucbvi":
        algo = UCBVI_CH_Tabular(
            mdp_on=mdp_on,
            cfg=cfg_alg,
        )
        return algo.run(rng_algo)

    algo = OfflineOnlineUCRL_TabularBlocks(
        mdp_on, cfg_alg, Vtilde, off_states, off_actions
    )
    return algo.run(rng_algo)


# ------------------------------------------------------
# 2) Core engine:
#    for each run i: fix mdp_on
#    for each sweep point j: generate mdp_off + offline data ONCE
#    for each algorithm: run with its own rng stream
# ------------------------------------------------------


def run_curves_many_fixed_online_for_sweep(
    base_cfg: SimConfig,
    x_values,
    make_cfg_from_x,          # (cfg_base, x) -> cfg_x  (may set Delta/M_off/z_action0/coverage_obs/etc)
    n_runs: int = 20,
    seed_offset: int = 0,
):
    settings = _get_algo_settings()
    x = np.asarray(x_values)
    curves_by_x = [{k: [] for k in settings.keys()} for _ in range(len(x))]

    for i in range(n_runs):
        seed_i = int(seed_offset + base_cfg.seed + i)

        # 1) FIX online MDP for this run i
        mdp_on_i = sample_online_mdp(
            base_cfg.if_shuffle,
            base_cfg.S,
            base_cfg.A,
            base_cfg.H,
            seed_i,
            base_cfg.s1,
        )

        for j, xj in enumerate(x):
            # 2) build cfg_x for this sweep point (Delta/M_off/z_action0/etc),
            #    but keep seed fixed to seed_i for this run.
            cfg_x = make_cfg_from_x(base_cfg, xj)
            cfg_x = SimConfig(**{**cfg_x.__dict__, "seed": seed_i})

            # 3) offline MDP from the SAME mdp_on_i, using cfg_x.
            mdp_off_ij = sample_offline_from_online1(
                mdp_on_i,
                cfg_x.if_shuffle,
                cfg_x.Delta,
                seed_i + 10_000 + 1000 * j,  # separate randomness per (i,j)
            )

            # 4) impose coverage structure consistently (creates new mdps; does not mutate mdp_on_i)

            mdp_on_ij = mdp_on_i
            behavior = RandomBehaviorPolicy(cfg_x.A)

            # 5) generate OFFLINE DATA ONCE per (i,j) using cfg_x (so M_off and z_action0 matter here)
            rng_off = np.random.default_rng(seed_i + 20_000 + 1000 * j)
            off_states, off_actions = collect_offline(mdp_off_ij, behavior, cfg_x, rng_off)

            # 6) run each algorithm:
            #    - cfg differs by algo (tildeV_mode/merge_mode/online_algo)
            #    - Vtilde must be generated inside the algo loop because tildeV_mode can differ
            for algo_name, kw in settings.items():
                cfg_ij_algo = SimConfig(**{**cfg_x.__dict__, **kw})

                # merge oracle: algorithm assumes Delta=0
                if cfg_ij_algo.merge_mode:
                    cfg_alg = SimConfig(**{**cfg_ij_algo.__dict__, "Delta": 0.0})
                else:
                    cfg_alg = cfg_ij_algo

                # Vtilde depends on tildeV_mode (per algorithm)
                rng_v = np.random.default_rng(
                    seed_i + 30_000 + 1000 * j + 17 * (abs(hash(algo_name)) % 10_000)
                )
                Vtilde = TildeVDesigner(cfg_x.S, cfg_x.A).design(cfg_alg.coverage_obs, cfg_alg.z_action0,
                    off_states, off_actions, cfg_ij_algo.tildeV_mode, rng_v
                )

                # stable per-(i,j,algo) RNG for the ONLINE interaction randomness
                rng_algo = np.random.default_rng(
                    seed_i + 40_000 + 1000 * j + 97 * (abs(hash(algo_name)) % 10_000)
                )

                reg = run_one_algo_given_everything(
                    cfg_alg=cfg_alg,
                    mdp_on=mdp_on_ij,
                    off_states=off_states,
                    off_actions=off_actions,
                    Vtilde=Vtilde,
                    rng_algo=rng_algo,
                )
                curves_by_x[j][algo_name].append(reg)

    # stack lists -> arrays
    for j in range(len(x)):
        for algo_name in curves_by_x[j]:
            curves_by_x[j][algo_name] = np.stack(curves_by_x[j][algo_name], axis=0)  # (n_runs, K)

    return x, curves_by_x



# ------------------------------------------------------
# 4) Sweeps (FIXED: mdp_on fixed per run i across x-series)
# ------------------------------------------------------

def run_and_plot_fixed_setting(
    cfg: SimConfig,
    n_runs: int = 20,
    seed_offset: int = 0,
    q_lo: float = 0.2,
    q_hi: float = 0.8,
    cumulative: bool = True,
    save_path: str = "regret_curves_fixed_setting.pdf",
    title: str = "",
    if_running: bool = True,
    save_loc: str = "curves_E.pkl",
):

    save_loc = Path(save_loc)
    if if_running:
        curves_runs = run_curves_many_fixed_setting(
            cfg=cfg,
            n_runs=n_runs,
            seed_offset=seed_offset,
        )

        with open(save_loc, "wb") as f:
            pickle.dump(curves_runs, f)

    else:
        with open(save_loc, "rb") as f:
            curves_runs = pickle.load(f)

    curves_runs = {k: v[:2] for k, v in curves_runs.items()}
    summary = summarize_curves(
        curves_runs,
        q_lo=q_lo,
        q_hi=q_hi,
        cumulative=cumulative,
    )

    plot_with_bands(
        summary,
        title=title,
        save_path=save_path,
    )

    return curves_runs, summary

def run_curves_many_fixed_setting(
    cfg: SimConfig,
    n_runs: int = 20,
    seed_offset: int = 0,
):
    def make_cfg(cfg0, _x):
        return cfg0

    _, curves_by_x = run_curves_many_fixed_online_for_sweep(
        base_cfg=cfg,
        x_values=[0],               # single point
        make_cfg_from_x=make_cfg,
        n_runs=n_runs,
        seed_offset=seed_offset,
    )
    return curves_by_x[0]


def sweep_over_Delta(base_cfg: SimConfig, Deltas,
                     n_runs: int = 20, seed_offset: int = 0,
                     q_lo=0.2, q_hi=0.8, cumulative: bool = True, if_running: bool = True, save_loc: str = 'curve_A.pkl'):
    """
    x = Deltas
    output:
      per_algo[name] = {"mean": (N,), "lo": (N,), "hi": (N,)}
    """

    def make_cfg(cfg0, Delta):
        return SimConfig(**{**cfg0.__dict__, "Delta": float(Delta)})

        save_loc = Path(save_loc)
    if if_running:
        x, curves_by_x = run_curves_many_fixed_online_for_sweep(
            base_cfg=base_cfg,
            x_values=Deltas,
            make_cfg_from_x=make_cfg,
            n_runs=n_runs,
            seed_offset=seed_offset,
        )

        with open(save_loc, "wb") as f:
            pickle.dump((x, curves_by_x), f)

    else:
        with open(save_loc, "rb") as f:
            x, curves_by_x = pickle.load(f)


    per_algo = {}
    for j in range(len(x)):
        final_ = summarize_final_at_K(curves_by_x[j], q_lo=q_lo, q_hi=q_hi, cumulative=cumulative)
        for name, stats in final_.items():
            per_algo.setdefault(name, {"mean": [], "lo": [], "hi": []})
            per_algo[name]["mean"].append(stats["mean"])
            per_algo[name]["lo"].append(stats["lo"])
            per_algo[name]["hi"].append(stats["hi"])

    for name in per_algo:
        for k in ["mean", "lo", "hi"]:
            per_algo[name][k] = np.asarray(per_algo[name][k], dtype=float)

    return np.asarray(x, dtype=float), per_algo


def sweep_over_Moff(base_cfg: SimConfig, Moffs,
                    n_runs: int = 20, seed_offset: int = 0,
                    q_lo=0.2, q_hi=0.8, cumulative: bool = True, if_running: bool = True, save_loc: str = 'curve_B.pkl'):

    def make_cfg(cfg0, M):
        return SimConfig(**{**cfg0.__dict__, "M_off": int(M)})


    if if_running:
        x, curves_by_x = run_curves_many_fixed_online_for_sweep(
            base_cfg=base_cfg,
            x_values=Moffs,
            make_cfg_from_x=make_cfg,
            n_runs=n_runs,
            seed_offset=seed_offset,
        )

        with open(save_loc, "wb") as f:
            pickle.dump((x, curves_by_x), f)

    else:
        with open(save_loc, "rb") as f:
            x, curves_by_x = pickle.load(f)

    per_algo = {}
    for j in range(len(x)):
        final_ = summarize_final_at_K(curves_by_x[j], q_lo=q_lo, q_hi=q_hi, cumulative=cumulative)
        for name, stats in final_.items():
            per_algo.setdefault(name, {"mean": [], "lo": [], "hi": []})
            per_algo[name]["mean"].append(stats["mean"])
            per_algo[name]["lo"].append(stats["lo"])
            per_algo[name]["hi"].append(stats["hi"])

    for name in per_algo:
        for k in ["mean", "lo", "hi"]:
            per_algo[name][k] = np.asarray(per_algo[name][k], dtype=float)

    return np.asarray(x, dtype=float), per_algo

def sweep_over_tau(
    base_cfg: SimConfig,
    taus,
    n_runs: int = 20,
    seed_offset: int = 0,
    q_lo=0.2,
    q_hi=0.8,
    cumulative: bool = True,
    if_running: bool = True,
    save_loc: str = 'curve_C.pkl'
):
    taus = np.asarray(taus, dtype=float)

    def make_cfg(cfg0, tau):
        z0 = float(tau)
        return SimConfig(**{
            **cfg0.__dict__,
            "coverage_obs": 0,
            "Delta": float(cfg0.Delta),
            "z_action0": z0,
        })

    if if_running:
        x, curves_by_x = run_curves_many_fixed_online_for_sweep(
            base_cfg=base_cfg,
            x_values=taus,
            make_cfg_from_x=make_cfg,
            n_runs=n_runs,
            seed_offset=seed_offset,
        )

        with open(save_loc, "wb") as f:
            pickle.dump((x, curves_by_x), f)

    else:
        with open(save_loc, "rb") as f:
            x, curves_by_x = pickle.load(f)


    per_algo = {}
    for j in range(len(x)):
        final_ = summarize_final_at_K(
            curves_by_x[j], q_lo=q_lo, q_hi=q_hi, cumulative=cumulative
        )
        for name, stats in final_.items():
            per_algo.setdefault(name, {"mean": [], "lo": [], "hi": []})
            per_algo[name]["mean"].append(stats["mean"])
            per_algo[name]["lo"].append(stats["lo"])
            per_algo[name]["hi"].append(stats["hi"])

    for name in per_algo:
        for k in ["mean", "lo", "hi"]:
            per_algo[name][k] = np.asarray(per_algo[name][k], dtype=float)

    return np.asarray(x, dtype=float), per_algo


def sweep_over_z_with_custom_Delta_curves_shared_online(
    base_cfg: SimConfig,
    z_series,
    Delta_series_list,     # list of arrays, each length N
    curve_labels=None,
    n_runs: int = 20,
    seed_offset: int = 0,
    q_lo=0.2,
    q_hi=0.8,
    cumulative: bool = True,
    if_running: bool = True,
    save_loc: str = 'curve_D.pkl'
):

    z = np.asarray(z_series, dtype=float)
    N = len(z)

    C = len(Delta_series_list)
    Delta_series_list = [np.asarray(Dc, dtype=float) for Dc in Delta_series_list]
    for Dc in Delta_series_list:
        assert len(Dc) == N, "Each Delta_series must have same length as z_series."

    if curve_labels is None:
        curve_labels = [f"curve_{c}" for c in range(C)]
    assert len(curve_labels) == C

    settings = _get_algo_settings()  # the algo settings dict


    if if_running:
        curves_runs = {
            curve_labels[c]: {algo: [[] for _ in range(N)] for algo in settings.keys()}
            for c in range(C)
        }
    
        for i in range(n_runs):
            seed_i = int(seed_offset + base_cfg.seed + i)
    
            # 1) FIX online MDP for this run i (shared across all curves and all z)
            mdp_on_i = sample_online_mdp(
                base_cfg.if_shuffle,
                base_cfg.S,
                base_cfg.A,
                base_cfg.H,
                seed_i,
                base_cfg.s1,
            )
    
            for j in range(N):
                z_j = float(z[j])
    
                for c in range(C):
                    Delta_cj = float(Delta_series_list[c][j])
                    label_c = curve_labels[c]
    
                    # 2) config for this (curve c, z index j), but seed fixed to seed_i
                    cfg_x = SimConfig(**{
                        **base_cfg.__dict__,
                        "seed": seed_i,
                        "coverage_obs": 0,
                        "z_action0": z_j,
                        "Delta": Delta_cj,
                    })
    
                    # 3) offline MDP from the SAME mdp_on_i
                    #    IMPORTANT: seed must include curve index c, otherwise offline may coincide across curves.
                    mdp_off_ijc = sample_offline_from_online1(
                        mdp_on_i,
                        cfg_x.if_shuffle,
                        cfg_x.Delta,
                        seed_i + 10_000 + 1_000 * j + 100_000 * c,
                    )
    
                    # 4) impose coverage structure (creates new mdps; does not mutate mdp_on_i)
                    mdp_on_ij = mdp_on_i
                    behavior = RandomBehaviorPolicy(cfg_x.A)
    
                    # 5) offline data ONCE per (i,j,c)
                    rng_off = np.random.default_rng(seed_i + 20_000 + 1_000 * j + 100_000 * c)
                    off_states, off_actions = collect_offline(mdp_off_ijc, behavior, cfg_x, rng_off)
    
                    # 6) run each algorithm on identical (mdp_on_ij, offline data)
                    for algo_name, kw in settings.items():
                        cfg_algo = SimConfig(**{**cfg_x.__dict__, **kw})
    
                        # merge oracle: algorithm assumes Delta=0
                        if cfg_algo.merge_mode:
                            cfg_alg = SimConfig(**{**cfg_algo.__dict__, "Delta": 0.0})
                        else:
                            cfg_alg = cfg_algo
    
                        # Vtilde depends on tildeV_mode (per algorithm)
                        rng_v = np.random.default_rng(
                            seed_i + 30_000 + 1_000 * j + 100_000 * c
                            + 17 * (abs(hash(algo_name)) % 10_000)
                        )
                        Vtilde = TildeVDesigner(cfg_x.S, cfg_x.A).design(
                            cfg_alg.coverage_obs,
                            cfg_alg.z_action0,
                            off_states,
                            off_actions,
                            cfg_algo.tildeV_mode,
                            rng_v,
                        )
    
                        # online interaction RNG (per algo)
                        rng_algo = np.random.default_rng(
                            seed_i + 40_000 + 1_000 * j + 100_000 * c
                            + 97 * (abs(hash(algo_name)) % 10_000)
                        )
    
                        reg_curve = run_one_algo_given_everything(
                            cfg_alg=cfg_alg,
                            mdp_on=mdp_on_ij,
                            off_states=off_states,
                            off_actions=off_actions,
                            Vtilde=Vtilde,
                            rng_algo=rng_algo,
                        )
    
                        if cumulative:
                            y = float(np.sum(reg_curve))  # cumulative regret at K
                        else:
                            y = float(reg_curve[-1])      # regret at K
    
                        curves_runs[label_c][algo_name][j].append(y)

        with open(save_loc, "wb") as f:
            pickle.dump(curves_runs, f)

    else:
        with open(save_loc, "rb") as f:
            curves_runs = pickle.load(f)

    # ---- finalize into curves_out[label][algo]={"mean","lo","hi"} arrays of length N ----
    curves_out = {}
    for label_c in curve_labels:
        per_algo = {}
        for algo_name in settings.keys():
            means, los, his = [], [], []
            for j in range(N):
                vals = np.asarray(curves_runs[label_c][algo_name][j], dtype=float)  # (n_runs,)
                means.append(float(np.mean(vals)))
                los.append(float(np.quantile(vals, q_lo)))
                his.append(float(np.quantile(vals, q_hi)))
            per_algo[algo_name] = {
                "mean": np.asarray(means, dtype=float),
                "lo":   np.asarray(los, dtype=float),
                "hi":   np.asarray(his, dtype=float),
            }
        curves_out[label_c] = per_algo

    return z, curves_out



def _plot_style_and_order():
    style = {
        "zero":        (r"\textsf{UCRL}",                         "o", "-"),
        "merge":       (r"\textsf{COMPLETE}",                     "P", "--"),
        "optimal":     (r"\textsf{O--O UCRL--VLR (Optimal)}",      "s", "-."),
        "pessimistic": (r"\textsf{O--O UCRL--VLR (pessimistic)}",  "^", ":"),
        "dp_ucb":      (r"\textsf{DP--LSVI}",                      "D", "--"),
        "ucbvi":       (r"\textsf{UCBVI}",                         "X", "-"),
    }
    order = ["zero", "merge", "dp_ucb", "ucbvi", "optimal", "pessimistic"]
    return style, order

