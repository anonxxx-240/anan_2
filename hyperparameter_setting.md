# Hyperparameter and Configuration Guide

This document describes the main configuration fields in `source_codes/hyperparam.py`.

## Overview

All experiments are controlled by the `SimConfig` dataclass. The current default configuration is designed for the synthetic tabular MDP experiments used to study offline–online learning under environment shift.

## Core fields in `SimConfig`

### Plot control

- `plt_choice_list = ['A', 'B', 'C', 'D', 'E', 'legend']`
  - Available figure targets.
- `plt_choice`
  - Chooses which panel to run.
  - Default in the current repo snapshot: `plt_choice_list[1]`, i.e. panel `B`.

### Repetition and execution mode

- `n_runs`
  - Number of independent Monte Carlo runs.
  - Default: `50`.
- `if_running`
  - Controls whether the code reruns simulations or loads cached results.
  - `1`: run simulations and overwrite cache.
  - `0`: load previously saved `.pkl` results when available.

### Random seed

- `seed`
  - Base random seed.
  - Default: `3`.

## Synthetic MDP size

- `S`
  - Number of states.
  - Default: `5`.
- `A`
  - Number of actions.
  - Default: `10`.
- `H`
  - Episode horizon.
  - Default: `3`.

These experiments use a tabular MDP that is treated through a linear-mixture embedding.

## Offline / online sample sizes

- `M_off`
  - Number of offline trajectories.
  - Default: `100000`.
- `K`
  - Number of online episodes.
  - Default: `4000`.

These two values are the main sample-size parameters in the paper and the simulations.

## Environment shift

- `Delta`
  - Environment-shift magnitude used by the algorithm.
  - Default: `0.05`.

In the code, `Delta` controls how different the offline environment is from the online environment.

- `if_shuffle`
  - Controls which offline-environment construction is used when generating a shifted offline MDP.
  - `0`: Gaussian perturbation plus simplex projection.
  - `1`: permutation-style or adversarial reshaping construction, depending on the helper used.

## Algorithmic parameters

- `lam`
  - Ridge regularization parameter.
  - Default: `1.0`.
- `delta_conf`
  - Confidence level for the estimators.
  - Default: `0.0`, which triggers automatic setting to `1 / K` in `__post_init__`.
- `B`
  - Norm bound used in the confidence radius.
  - Default: `2.0`.
- `bonus_scale`
  - Extra scaling used in some optimistic bonuses.
  - Default: `0.02`.

## Offline value-function design

- `tildeV_mode`
  - Controls the construction of offline value-function estimates.
  - Supported modes in the current code/comments:
    - `"optimal"`
    - `"zero"`
    - `"random"`
    - `"pessimistic"` is also used by the runner settings in `prep.py`
  - Default: `"optimal"`.

This is one of the most important experimental choices in the repo, since the paper studies how the design of offline value estimates affects coverage and regret.

## Coverage setting

- `coverage_obs`
  - Selects how the offline behavior policy is generated.
  - `1`: random offline policy.
  - `0`: coverage-controlled policy with a two-region structure and a biased probability on action `0`.
  - Default: `1`.

- `z_action0`
  - Bias parameter used in the coverage-controlled offline policy when `coverage_obs = 0`.
  - Default input: `0.0`.
  - Actual default after initialization: `0.3`.

Interpretation:

- Larger `z_action0` increases the frequency of action `0` in one region of the state space.
- This changes the offline coverage geometry and therefore affects the effective coverage parameter `tau` studied in the paper.

## Merge mode

- `merge_mode`
  - If `True`, the algorithm assumes `Delta = 0` in the estimator combination step, so the shift term disappears.
  - Default: `False`.

This is mainly useful for testing naive merge-style behavior and comparing against methods that ignore environment shift.

## Value clipping

- `clip_value`
  - Whether to clip optimistic value estimates.
  - Default: `True`.

## Online algorithm selector

- `online_algo`
  - Chooses the online algorithm family.
  - Default: `"ours"`.
  - Other names appear in the runners, such as:
    - `"dp_ucb_t2"`
    - `"ucbvi"`

## Initial state

- `s1`
  - Initial state index.
  - Default: `0`.

## Automatic settings in `__post_init__`

The following values are automatically adjusted when left nonpositive:

- If `delta_conf <= 0`, set
  - `delta_conf = 1 / max(1, K)`.
- If `z_action0 <= 0`, set
  - `z_action0 = 0.3`.

## Figure-specific notes

### Panel A: final regret vs `Delta`

- Uses a sweep over a list of `Delta` values.
- Keeps `coverage_obs = 1`.

### Panel B: final regret vs `M_off`

- Uses a sweep over offline sample size.
- Fixes `Delta = 0.1` in the current code.

### Panel C: final regret vs `tau`

- Implemented through a sweep over `z_action0` under `coverage_obs = 0`.
- In the current experiment code, `tau` is represented operationally by this coverage-control parameter.

### Panel D: custom `Delta(tau)` curves

- Uses several manually specified growth rules for `Delta` as a function of `tau`, including linear, quadratic, and cubic scaling.
- Intended to illustrate the theoretical prediction that offline data remains informative when `Delta / tau` stays controlled.

### Panel E: regret curves over `K`

- Runs a fixed setting and plots cumulative regret across online episodes.

## Suggested starting points

### Reproduce the main qualitative comparisons

Use the default values in `hyperparam.py`, then set:

- `plt_choice = 'A'` for regret vs `Delta`
- `plt_choice = 'B'` for regret vs `M_off`
- `plt_choice = 'C'` for regret vs `tau`
- `plt_choice = 'D'` for the `Delta(tau)` scaling experiment
- `plt_choice = 'E'` for regret curves over `K`

### Study robustness to environment shift

- Increase `Delta` while keeping `coverage_obs = 1`.
- Compare the proposed algorithm with the online-only baseline.

### Study coverage effects

- Set `coverage_obs = 0`.
- Sweep `z_action0`.
- Compare `tildeV_mode = "optimal"` against `"pessimistic"` or `"zero"`.

## Practical notes

- The plotting code saves figures in the current working directory.
- Some sweep functions cache simulation outputs as `.pkl` files.
- If you change the configuration substantially, delete stale cache files or rerun with `if_running = 1`.
