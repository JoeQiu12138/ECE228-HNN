# Phase 2 HHsystem

Hénon–Heiles Hamiltonian neural network (HNN) for Phase 2.

- **Recommended pipeline:** `hh_train_eval.py` + `hh_model.py` (CPU or CUDA).
- **Legacy reference:** `legacy_henon_heiles_train.py` (original single-file script; edit `actF` manually).

All activation **figures**, **numeric txt**, and the historical **`data/` archive** are under [`exp_data/HH/`](../../exp_data/HH/) (`HHsystem/data/` was moved to `exp_data/HH/data/`). This folder holds **code, checkpoints, and tables only**.

---

## Quick start

Run all commands from this directory:

```bash
cd src/Phase2/HHsystem

# Sanity: short train + eval on CPU and CUDA (if available)
python3 hh_train_eval.py --smoke

# Retrain everything (long + short + polynomial probe)
python3 hh_train_eval.py --device cuda --train all

# Retrain specific runs
python3 hh_train_eval.py --device cuda --train mysin-long gabor-long

# Regenerate comparison table (metrics from checkpoint exports under exp_data/HH/)
python3 hh_train_eval.py --device auto --eval-only

# Gabor long: score from exp_data/HH/data/ archive instead of your checkpoint export
python3 hh_train_eval.py --eval-only --golden-gabor

# Export PNG + txt for every checkpoint
python3 hh_train_eval.py --export-all --device cuda

# Fast pipeline test
python3 hh_train_eval.py --device cpu --quick --train mysin-long
```

Device notes and GPU utilization: [`WORKFLOW.md`](WORKFLOW.md).

---

## Scripts

### `hh_train_eval.py` — main entry point

Builds `results/hh_activation_comparison.{md,csv,json}`, trains missing activations, exports artifacts to `exp_data/HH/`, and syncs canonical Gabor numerics from `exp_data/HH/data/` into `Gabor/longtime/`.

| Flag | Purpose |
| --- | --- |
| `--smoke` | Short train/eval on CPU and CUDA; exit (no full table) |
| `--eval-only` | Evaluate existing checkpoints; write comparison table |
| `--train TARGET …` | Train listed targets (`mysin-long`, `gabor-short`, `all`, …) |
| `--golden-gabor` | With eval: sync/score Gabor long from `exp_data/HH/data/` archive |
| `--export-all` | Export plots/txt for every checkpoint (no training) |
| `--device {auto,cuda,cpu}` | `auto` = CUDA if available, else CPU |
| `--quick` | Use reduced epoch counts for a fast end-to-end test |
| `--short-epochs N` / `--long-epochs N` | Override stage epoch counts |
| `--log-every N` | Print training status every N epochs |

**Typical workflows**

```bash
# After cloning: verify environment
python3 hh_train_eval.py --smoke

# Refresh the report table from existing ckpts + experiment txt
python3 hh_train_eval.py --eval-only --device auto

# Retrain selected runs, then evaluate
python3 hh_train_eval.py --device cuda --train mysin-long adaptivesin-long
python3 hh_train_eval.py --eval-only --device auto

# Re-export all figures from checkpoints
python3 hh_train_eval.py --export-all --device cuda
python3 hh_train_eval.py --eval-only --device auto
python3 hh_train_eval.py --eval-only --golden-gabor   # optional historical Gabor row
```

**Outputs:** `results/hh_activation_comparison.*`, updates under `exp_data/HH/<activation>/`.

---

### `hh_model.py` — shared model and training

Imported by `hh_train_eval.py`. Defines the HNN, five activations, Hamiltonian loss, `train_hh_model`, and `evaluate_hh_model`. Not meant to be run directly.

```python
# Example (from another script in this folder):
from hh_model import train_hh_model, evaluate_hh_model, ACTIVATIONS
```

---

### `hh_physics_utils.py` — ground truth and integrators

HH exact solution hooks, `odeint` reference trajectories, symplectic Euler (`symEuler`), and energy. Imported by `hh_model.py` and `hh_train_eval.py`.

```python
from hh_physics_utils import HHsolution, energy
```

---

### `legacy_henon_heiles_train.py` — original monolithic script

Reproduces the author’s manual workflow: set `actF` in the model class, uncomment **stage 1 (short)** or **stage 2 (long)** at the bottom, then run:

```bash
python3 legacy_henon_heiles_train.py
```

- Default long-time run loads `models/model_HH_Gabor_longtime.zip` and writes loss to `exp_data/HH/data/loss.txt`.
- Figures are saved next to the script unless paths were changed in the file.
- Prefer `hh_train_eval.py` for multi-activation comparison and reproducible tables.

---

## Directories

| Path | Role |
| --- | --- |
| `models/` | Checkpoints `model_HH_<activation>_<stage>.zip` |
| `results/` | `hh_activation_comparison.md` / `.csv` / `.json` |
| `exp_data/HH/` (repo root) | All run artifacts (see [`exp_data/HH/README.md`](../../exp_data/HH/README.md)) |

---

## Checkpoint naming

| File | Activation | Stage |
| --- | --- | --- |
| `model_HH_Gabor_longtime.zip` | GaborActivation | long |
| `model_HH_Gabor_short_time.zip` | GaborActivation | short |
| `model_HH_adaptiveSin_shorttime.zip` | AdaptiveSin | short |
| `model_HH_adaptiveSin_longtime.zip` | AdaptiveSin | long |
| `model_HH_mySin_short.zip` / `model_HH_mySin_longtime.zip` | mySin | short / long |
| `model_HH_dualSin_shorttime.zip` / `model_HH_dualSin_longtime.zip` | DualAdaptiveSin | short / long |
| `model_HH_polynomial_probe.zip` | LearnablePolynomial | probe only |

Legacy names `model_HH.zip` and `model_HH_short.zip` are renamed automatically the first time you run `hh_train_eval.py`.

---

## Data flow

```
models/*.zip          ──train/export──►  exp_data/HH/<activation>/
exp_data/HH/data/   ──sync──────────►  exp_data/HH/Gabor/longtime/
results/*.md          ◄──eval──────────  exp_data/HH/*/dx.txt
```

---

## Do not delete

- `legacy_henon_heiles_train.py` — reference for original Gabor training schedule  
- `exp_data/HH/data/` — historical Gabor numerics archive  
- `models/model_HH_Gabor_longtime.zip` — long-time Gabor weights  
