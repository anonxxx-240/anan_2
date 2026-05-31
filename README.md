# Offline–Online Reinforcement Learning for Linear Mixture MDPs

## Citation

If you use this code or build on this work, please cite:
```bibtex
@article{zhang2026offline,
  title={Offline-Online Reinforcement Learning for Linear Mixture MDPs},
  author={Zhang, Zhongjun and Sinclair, Sean R},
  journal={arXiv preprint arXiv:2604.11994},
  year={2026}
}
```

---

## 1. Overview

This repository contains all code required to reproduce the numerical experiments and figures for the paper *Offline–Online Reinforcement Learning for Linear Mixture MDPs*.

The codebase studies offline–online reinforcement learning under environment shift and unknown behavior policies. It implements synthetic tabular MDP environments with linear-mixture structure, evaluates regret under varying conditions, and generates all figures reported in the study.

A typical replication workflow is:

1. Set up the computational environment (Section 4)
2. Configure and run simulations (Section 5)
3. Generate plots from cached or newly computed results

---

## 2. Data and experimental setting

This repository uses **synthetically generated data only**.

### Synthetic environments

All experiments are conducted on tabular MDPs constructed using a linear-mixture representation. The environments, transition dynamics, and reward structures are generated programmatically within the codebase.

Offline datasets are generated using predefined behavior policies, with key parameters including:

* `M_off`: number of offline samples
* `tau`: effective coverage of offline data
* `Delta`: environment shift between offline and online phases

No external datasets are required.

---

## 3. Variable definitions

Below are the main variables used throughout the experiments:

### Core parameters

* `Delta`: magnitude of environment shift
* `M_off`: size of offline dataset
* `tau`: coverage parameter for offline data
* `K`: number of online episodes
* `seed`: random seed for reproducibility

### Performance metrics

* `regret`: cumulative or final regret under the online policy
* `final_regret`: regret evaluated at the end of learning

---

## 4. Computational requirements

All code is written in Python (**>= 3.10**).

### Dependencies

Key dependencies include:

* numpy
* matplotlib
* pickle (for caching results)

All required packages are listed in:

```text id="t7wq6v"
requirements.txt
```

### Environment setup

```bash id="9t3b6g"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Runtime notes

* Simulation sweeps may take tens of hours (under $50$ runs) depending on parameter settings
* Cached `.pkl` files are used to avoid recomputation
* Random seeds are fixed where appropriate for reproducibility

---

## 5. Programs / Code

### Main execution

* `source_codes/main.py` — Entry point for all experiments
* `source_codes/hyperparam.py` — Central configuration (`SimConfig`)

### Experiment pipeline

* `source_codes/prep.py` — Runs simulations, parameter sweeps, and caching
* `source_codes/plot.py` — Generates figures from simulation outputs

### Environment and data generation

* `source_codes/MDP.py` — Linear-mixture MDP construction
* `source_codes/behavior_policy.py` — Offline data generation policies

### Algorithms

* `source_codes/algo.py` — Main offline–online algorithm
* `source_codes/algo_chen2022.py` — Baseline method
* `source_codes/Azar17.py` — UCBVI-style online baseline

### Running experiments

All experiments are controlled via:

```bash id="6j2n3x"
python source_codes/main.py
```

The behavior is determined by `SimConfig` in:

```text id="c0o3ff"
source_codes/hyperparam.py
```

Key configuration:

* `plt_choice`: selects which figure to generate

The code supports the following figure options through `SimConfig.plt_choice`:

- `"A"`: final regret versus `Delta`
- `"B"`: final regret versus `M_off`
- `"C"`: final regret versus `tau`
- `"D"`: final regret versus `tau` under custom `Delta(tau)` curves
- `"E"`: regret curves over online episodes `K`
- `"legend"`: legend-only export for figure assembly
* `if_running`:

  * `1` → run simulations
  * `0` → load cached results

### Output

Figures are saved in the working directory, including:

* `final_regret_vs_Delta.pdf`
* `final_regret_vs_Moff.pdf`
* `final_regret_vs_tau_fixed_Delta.pdf`
* `regret_curves_random_offline.pdf`

Cached intermediate results are stored as `.pkl` files.

## 6. Example workflow

1. Open `source_codes/hyperparam.py`.
2. Set `plt_choice` to the panel you want.
3. Set `if_running = 1` to run simulations and cache results.
4. Run:

```bash
python source_codes/main.py
```

If cached result files already exist, setting `if_running = 0` reloads them instead of rerunning the simulations.

## 7. Baseline algorithms included

The simulation pipeline compares the proposed method against several baselines, including:

- `zero`: online-only / no useful offline value design
- `optimal`: proposed offline value-function design
- `pessimistic`: pessimistic offline value-function design
- `merge`: naive merge baseline that effectively assumes no environment shift
- `dp_ucb`: tabular baseline from the Chen et al. style implementation
- `ucbvi`: tabular online baseline

---

All scripts use relative paths and can be executed independently once the environment is set up. Running the simulation pipeline reproduces all reported experimental results.
