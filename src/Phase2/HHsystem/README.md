# Phase 2 — Hénon–Heiles

**Entry:** `hh_train_eval.py`, `hh_model.py`, `hh_physics_utils.py`  
**Artifacts:** `exp_data/HH/`  
**Table:** [`results/hh_activation_comparison.md`](results/hh_activation_comparison.md)  
**Supplementary:** [`studies/README.md`](studies/README.md) → `exp_data/HH_studies/`

---

## Commands

Run from `src/Phase2/HHsystem/`:

| Goal | Command |
| --- | --- |
| Sanity (CPU + CUDA) | `python3 hh_train_eval.py --smoke` |
| Comparison table | `python3 hh_train_eval.py --device auto --eval-only` |
| Retrain all | `python3 hh_train_eval.py --device cuda --train all` |
| Retrain one | `python3 hh_train_eval.py --device cuda --train gabor-long` |
| Export figures/txt | `python3 hh_train_eval.py --export-all --device cuda` |
| Historical Gabor row | `python3 hh_train_eval.py --eval-only --golden-gabor` |
| Fast test | `python3 hh_train_eval.py --device cpu --quick --train mysin-long` |

**Train targets:** `mysin-long`, `adaptivesin-long`, `gabor-long`, `dualsin-long`, `polynomial-probe`, `all`, etc. (short aliases: `mysin` → `mysin-long`, `gabor` → `gabor-long`).

| Flag | Description |
| --- | --- |
| `--eval-only` | Evaluate checkpoints → `results/` table |
| `--train TARGET …` | Retrain (overwrites checkpoints) |
| `--golden-gabor` | Gabor long: copy `exp_data/HH/data/` → `Gabor/longtime/`, score archive |
| `--export-all` | Export PNG/txt from all checkpoints |
| `--device auto\|cuda\|cpu` | Compute device |
| `--quick` | Shorter epochs |

Default (no flags) = `--eval-only` without Gabor archive sync.

**Gabor:** normal eval uses **your** checkpoint exports in `Gabor/longtime/`. `--golden-gabor` uses author-era txt in `exp_data/HH/data/`.

**GPU ~10–20%:** small net + heavy Hamiltonian autograd per epoch; normal for this codebase.

---

## Layout

| Path | Role |
| --- | --- |
| `models/` | `model_HH_<activation>_<stage>.zip` |
| `results/` | Comparison tables |
| `studies/` | Energy tiers + Poincaré |
| `legacy_henon_heiles_train.py` | Original manual script |

---

## Protocol & best row

$X_0=[0,0.3,-0.3,0.3,0.15,1]$, $E_0\approx0.128$, $t\in[0,12\pi]$, 500 points, 80 neurons, short 20k + long 50k epochs.

**Gabor long (May 2026):** max traj. err **1.586e-03**, max E drift **6.022e-04**.
