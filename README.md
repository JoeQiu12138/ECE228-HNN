# Physics-Informed Hamiltonian Neural Networks for Chaotic Systems
**University of California, San Diego - ECE 228 Final Project**

This repository contains the implementation, experiments, and ablation studies for exploring **Hamiltonian-equation-informed neural solvers inspired by Hamiltonian Neural Networks (HNNs)**. The primary objective is to learn long-time solution trajectories for the highly non-linear and chaotic **Hénon-Heiles (HH) system**, and to investigate how physical constraints and network architectures affect long-time predictions in chaotic regimes.

---

## 📌 Project Overview
Traditional data-driven deep learning models often fail to capture the long-term dynamics of chaotic physical systems, suffering from energy drift and trajectory divergence over time. 

Instead of treating the trajectory prediction as a standard regression task, we use a **physics-informed neural solution ansatz** and enforce Hamilton's equations via PyTorch's Autograd mechanism during training. The loss combines Hamiltonian-equation residuals with an energy-conservation penalty, encouraging trajectories to remain on the correct physical manifold and reducing long-time energy drift.

Our primary testbed is the **2D Hénon-Heiles System**, a classic model for Hamiltonian chaos with a strict escape energy threshold ($E=1/6$).

---

## Repository layout

```
ECE228-HNN/
├── src/           # Phase 1–3 source code
├── exp_data/      # experiment outputs (1D/, HH/)
└── reference/     # PDF papers and lecture notes
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Experimental Methodology & Physics Diagnosis

Our research was conducted in progressive stages, starting from a simple baseline and scaling up to chaotic environments. Instead of blindly tuning hyperparameters, our experiments focused on **opening the black box of network architectures** to understand how physical gradients flow through different activation layers.

### Phase 1: Proof of Concept on 1D Non-Linear Oscillator
Before tackling chaos, we first evaluated the network's foundational capacity on a simpler 1D non-linear oscillator.
* **Objective:** Compare standard deep learning activation functions (e.g., `GELU`, `Tanh`) against periodic/physics-informed activations (e.g., `AdaptiveSin`, `Snake`, baseline `mySin`).
* **Insight:** Traditional activation functions commonly used in CV/NLP performed poorly in learning conservative dynamics. However, adaptive and purely periodic functions (`AdaptiveSin`, `mySin`) demonstrated superior performance. This confirmed our initial hypothesis: **global periodicity and smooth bounded derivatives are essential for modeling conservative physical fields.**

**Phase 1 experiment settings.**

| Item | Value |
| --- | --- |
| System | 1D nonlinear oscillator |
| Initial state | $x_0=1.3$, $p_0=1.0$, $\lambda=1$ |
| Long-time rollout | $t \in [0, 20\pi]$ |
| Training points | 500 |
| Hidden width | 80 neurons |
| Epochs | 50,000 |
| Learning rate | $8\times10^{-3}$ |
| Baselines compared | `mySin`, `AdaptiveSin`, `Snake`, `GELU`, `Tanh` |

**Current saved Phase 1 quantitative summary.**

| Run | Final training loss | Max trajectory error | Mean trajectory error | Max energy drift | Mean energy drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1D nonlinear oscillator, current saved model | $1.28\times10^{-4}$ | $6.39\times10^{-3}$ | $1.92\times10^{-3}$ | $4.11\times10^{-3}$ | $1.03\times10^{-3}$ |

Additional activation-specific figures are stored under `exp_data/1D/`. Figures for `Tanh` are under `exp_data/1D/tanh/`.

### Phase 2: Scaling to the 2D Chaotic Hénon-Heiles System
Equipped with the insights from Phase 1, we scaled our model to the highly complex, 2D chaotic Hénon-Heiles system. Here, the challenge lies in the non-linear coupling terms that cause unpredictable chaotic trajectories. We conducted extensive ablation studies by designing custom, highly interpretable activation functions to see if structural priors could help the network:
* **Learnable Polynomials:** We designed a 3rd-order polynomial activation to perfectly match the theoretical HH potential equation. 
  * *Diagnosis:* This caused catastrophic gradient explosions. The composition of polynomials in a deep MLP expands the order exponentially, leading to numerical instability when computing physical forces via backpropagation.
* **Gabor Wavelets (Localized Resonances):** We utilized a Gaussian envelope multiplied by a sine wave to capture the system's local transient resonances.
  * *Diagnosis:* The network forced the Gaussian envelope to become extremely narrow, causing "tunnel vision" (dead gradients) at the high-energy edges of the phase space, failing to capture global chaotic movements.
* **Dual Adaptive Sine:** We introduced multiple learnable frequencies aiming to capture the beat frequencies of the non-linear coupling.
  * *Diagnosis:* The network naturally decayed the weight of the secondary frequency, collapsing back into a single dominant frequency.

**Phase 2 experiment settings.**

| Item | Value |
| --- | --- |
| System | 2D Hénon-Heiles system |
| Initial state | $x_0=0.3$, $y_0=-0.3$, $p_{x0}=0.3$, $p_{y0}=0.15$, $\lambda=1$ |
| Initial energy | $E_0=0.12825$ |
| Escape threshold | $E=1/6\approx0.16667$ |
| Regime represented by current run | Low-energy bounded HH trajectory below escape threshold |
| Long-time rollout | $t \in [0, 12\pi]$ |
| Training points | 500 |
| Hidden width | 80 neurons |
| Epochs | 50,000 for long-time continuation |
| Learning rate | $5\times10^{-3}$ |
| Baselines compared | `mySin`, `AdaptiveSin`, `DualAdaptiveSin`, `GaborActivation`, `LearnablePolynomial` |

**Current saved Phase 2 quantitative summary.**

| Run | Final training loss | Max trajectory error | Mean trajectory error | Max energy drift | Mean energy drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| HH system, current saved model (GaborActivation, long) | $8.79\times10^{-4}$ | $1.44\times10^{-3}$ | $3.89\times10^{-4}$ | $6.02\times10^{-4}$ | $1.75\times10^{-4}$ |

**Full HH activation comparison table (all ablations):** see [`src/Phase2/HHsystem/results/hh_activation_comparison.md`](src/Phase2/HHsystem/results/hh_activation_comparison.md). Regenerate with `python3 hh_train_eval.py` from `src/Phase2/HHsystem/` (see [`src/Phase2/HHsystem/README.md`](src/Phase2/HHsystem/README.md) and [`WORKFLOW.md`](src/Phase2/HHsystem/WORKFLOW.md)).

**Activation ablation artifact map.**

| System | Activation / variant | Available artifacts | Interpretation to use in report |
| --- | --- | --- | --- |
| 1D oscillator | `mySin` baseline | `exp_data/1D/baseline/` | Periodic baseline for conservative dynamics |
| 1D oscillator | `AdaptiveSin` | `exp_data/1D/adaptiveSin/` | Learnable frequency improves the periodic inductive bias |
| 1D oscillator | `Snake` | `exp_data/1D/Snake/` | Periodic residual activation; useful comparison against pure sine |
| 1D oscillator | `GELU` | `exp_data/1D/GELU/` | Standard smooth ML activation baseline |
| 1D oscillator | `Tanh` | `exp_data/1D/tanh/` | Standard bounded activation baseline |
| HH system | `mySin` baseline | `exp_data/HH/baseline/` | Baseline HH physics-informed neural solver |
| HH system | `AdaptiveSin` | `exp_data/HH/adaptivesin/a=1 longtime/` | Single learnable-frequency periodic prior |
| HH system | `DualAdaptiveSin` | `exp_data/HH/dualSin/longtime/` | Tests whether two learnable frequencies remain useful in chaotic coupling |
| HH system | `GaborActivation` | `exp_data/HH/Gabor/short/`, `exp_data/HH/Gabor/longtime/` | Localized oscillatory prior; useful for diagnosing local-vs-global dynamics |

The current implementation switches activations manually inside the model class. For reproducible final-report runs, record the active activation, seed, rollout horizon, hidden width, epoch count, and learning rate alongside each generated figure.
---


## ⚙️ How to Run

### Requirements
* Python 3.8+
* PyTorch
* NumPy
* Matplotlib

### Phase 1 — 1D nonlinear oscillator

```bash
cd src/Phase1/NLoscillator
python3 HNN_NLoscillator.py
```

Edit `actF` in the model class for each activation. All artifacts live under `exp_data/1D/` (figures per activation, numerics in `exp_data/1D/data/`). See [`src/Phase1/README.md`](src/Phase1/README.md), [`exp_data/1D/README.md`](exp_data/1D/README.md).

### Phase 2 — Hénon–Heiles (recommended)

From the repo root:

```bash
cd src/Phase2/HHsystem
python3 hh_train_eval.py --smoke                    # CPU/CUDA sanity check
python3 hh_train_eval.py --eval-only --device auto  # comparison table only
python3 hh_train_eval.py --device cuda --train all
python3 hh_train_eval.py --eval-only --golden-gabor   # optional: historical Gabor archive row
```

- **Main pipeline:** `hh_train_eval.py` (training, evaluation, table, export to `exp_data/HH/`)
- **Shared library:** `hh_model.py`, `hh_physics_utils.py`
- **Legacy single-activation script:** `legacy_henon_heiles_train.py` (edit `actF` and stage blocks at the bottom)

Full CLI and paths: [`src/Phase2/HHsystem/README.md`](src/Phase2/HHsystem/README.md), [`src/Phase2/HHsystem/WORKFLOW.md`](src/Phase2/HHsystem/WORKFLOW.md).  
Artifact index: [`exp_data/HH/README.md`](exp_data/HH/README.md).

### Phase 2 — legacy monolithic script (optional)

```bash
cd src/Phase2/HHsystem
python3 legacy_henon_heiles_train.py
```

Loads initial conditions, trains the HNN for the active activation, and writes loss/figures (see comments in the file for short vs long stages).

### Phase 3 — Coupled oscillators (cross-system validation)

```bash
cd src/Phase3/coupled_oscillator
python3 p3_train_eval.py --device cuda --train all
python3 p3_train_eval.py --eval-only --device auto
```

Cross-system table: `src/Phase3/coupled_oscillator/results/phase2_vs_phase3_comparison.md`.  
See [`src/Phase3/README.md`](src/Phase3/README.md).

---
*Disclaimer: This project builds upon the theoretical framework of [Hamiltonian Neural Networks (Greydanus et al., 2019)] and specific implementations from [Mattheakis et al. (Phys. Rev. E, 2022)].*
