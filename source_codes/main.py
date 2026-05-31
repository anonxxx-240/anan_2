from prep import *
from plot import *
from hyperparam import *

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base = SimConfig()

    n_runs = base.n_runs

    if base.plt_choice == 'A':
        # =========================
        # NEW Fig A: x = Delta, y = cumulative regret at K
        # =========================
        Deltas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 5, 10]
        cfgA = SimConfig(**{**base.__dict__, "coverage_obs": 1})
        xA, outA = sweep_over_Delta(cfgA, Deltas, n_runs=n_runs, seed_offset=200000, if_running = base.if_running)
        plot_final_vs_x(
            xA, outA,
            xlabel=r"$\Delta$",
            ylabel=r"\textsf{Cumulative Regret at }$K$",
            title="Final regret vs Delta",
            save_path="final_regret_vs_Delta.pdf",
        )

    if base.plt_choice == 'B':
        # =========================
        # # NEW Fig B: x = M_off, y = cumulative regret at K
        # # =========================
        Moffs = [100, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 500000]
        cfgB = SimConfig(**{**base.__dict__, "coverage_obs": 1, "Delta": 0.1})
        xB, outB = sweep_over_Moff(cfgB, Moffs, n_runs=n_runs, seed_offset=300000, if_running = base.if_running)
        plot_final_vs_x(
            xB, outB,
            xlabel=r"$M_{\mathrm{off}}$",
            ylabel=r"\textsf{Cumulative Regret at }$K$",
            title="Final regret vs M_off",
            save_path="final_regret_vs_Moff.pdf",
        )

    if base.plt_choice == 'C':
        # # =========================
        # # NEW Fig C ranges over \tau
        # # =========================
        taus = np.array([0.01, 0.025, 0.05, 0.1,  0.25, 0.5, 1.0], dtype=float)
        base.M_off = 200000

        xT, outT = sweep_over_tau(
            base_cfg=SimConfig(**{**base.__dict__, "coverage_obs": 0}),
            taus=taus,
            n_runs=n_runs,
            seed_offset=700000,
            cumulative=True,
            if_running = base.if_running
        )

        plot_final_vs_x(
            xT,
            outT,
            xlabel=r"$\tau$",
            ylabel=r"\textsf{Cumulative Regret at }$K$",
            title=rf"Final regret vs $\tau$",
            save_path="final_regret_vs_tau_fixed_Delta.pdf",
        )

    if base.plt_choice == 'D':
        # # =========================
        # # NEW Fig D (coverage_obs=0): x = Delta / z_action0
        # # =========================

        z_series = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0], dtype=float)
        Delta_series_list = [
            (1 / 200 * (100 * z_series) ** 1),
            (1 / 200 * (100 * z_series) ** 2),
            (1 / 200 * (100 * z_series) ** 3),
        ]
        curve_labels = [
            r"$\Delta(\tau)=0.5\tau$",
            r"$\Delta(\tau)=\frac{1}{200}(100\tau)^2$",
            r"$\Delta(\tau)=\frac{1}{200}(100\tau)^3$"
        ]
        cfgC = SimConfig(**{**base.__dict__, "coverage_obs": 0})
        z_out, curves_out = sweep_over_z_with_custom_Delta_curves_shared_online(
            base_cfg=cfgC,
            z_series=z_series,
            Delta_series_list=Delta_series_list,
            curve_labels=curve_labels,
            n_runs=n_runs,
            seed_offset=400000,
            q_lo=0.2,
            q_hi=0.8,
            cumulative=True,  # cumulative regret at K
            if_running = base.if_running
        )
        plot_two_algos_multi_curves(
            z=z_out,
            curves_out=curves_out,
            algo_keys=("zero", "optimal"),
            xlabel=r"$\tau$",
            ylabel=r"\textsf{Regret}$(K)$",
            save_path="final_regret_vs_z_multi_curves_zero_optimal.pdf",
            xlog=True,
            ylog=False,
        )

    if base.plt_choice == 'E':
        # =========================
        # NEW Fig E: ranges over K
        # =========================
        cfg_rand = SimConfig(**{**base.__dict__, "coverage_obs": 1})
        cfg_rand.K = 10000

        run_and_plot_fixed_setting(
            cfg=cfg_rand,
            n_runs=n_runs,
            seed_offset=0,
            cumulative=True,  # x=K, y=cum regret
            save_path="regret_curves_random_offline.pdf",
            title=f"coverage_obs=1, n_runs={n_runs}",
            if_running = base.if_running
        )

    if base.plt_choice == 'legend':
        save_legend_only(
        save_path="figure2legend.pdf",
        algo_subset=["zero","merge","dp_ucb","ucbvi","optimal","pessimistic"],
        ncol=3,
)
