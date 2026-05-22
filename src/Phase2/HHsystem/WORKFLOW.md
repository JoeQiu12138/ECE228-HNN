# Phase 2 HH — CPU / GPU workflow

Overview: [`README.md`](README.md).  
Training/eval: **`hh_train_eval.py`** + **`hh_model.py`** + **`hh_physics_utils.py`**.

## Command cheat sheet

Run from `src/Phase2/HHsystem/`:

| Goal | Command |
| --- | --- |
| Sanity (CPU + CUDA) | `python3 hh_train_eval.py --smoke` |
| Comparison table only | `python3 hh_train_eval.py --device auto --eval-only` |
| Retrain all activations | `python3 hh_train_eval.py --device cuda --train all` |
| Retrain one activation | `python3 hh_train_eval.py --device cuda --train gabor-long` |
| Historical Gabor metrics | `python3 hh_train_eval.py --eval-only --golden-gabor` |
| Export all figures/txt | `python3 hh_train_eval.py --export-all --device cuda` |
| Fast test | `python3 hh_train_eval.py --device cpu --quick --train mysin-long` |

### Train targets (`--train`)

`mysin-long`, `adaptivesin-long`, `adaptivesin-short`, `dualsin-long`, `dualsin-short`, `gabor-long`, `gabor-short`, `polynomial-probe`, or `all`.

Short aliases: `mysin` → `mysin-long`, `gabor` → `gabor-long`, `polynomial` → `polynomial-probe`.

### Flags

| Flag | Description |
| --- | --- |
| `--eval-only` | No training; evaluate checkpoints and write `results/` table |
| `--train TARGET …` | Retrain selected runs (always overwrites checkpoints) |
| `--golden-gabor` | Gabor long: copy `exp_data/HH/data/` → `Gabor/longtime/`, score archive |
| `--export-all` | Export PNG/txt from every existing checkpoint |
| `--device auto\|cuda\|cpu` | Compute device |
| `--quick` | Shorter epoch counts |

**Default:** no flags = same as `--eval-only` (no Gabor archive sync).

## Gabor: checkpoint vs golden archive

- **Normal eval** uses `exp_data/HH/Gabor/longtime/` txt produced by **your** checkpoint (`--export-all` or training export).
- **`--golden-gabor`** uses the author-era numerics in `exp_data/HH/data/` for the comparison table (use only when you explicitly want that reference row).

## GPU utilization (~10–20%)

Small net + fourth-order Hamiltonian autograd per epoch; sequential Adam. Parallelize by running different `--train TARGET` jobs in separate terminals.

## Artifacts

All under `exp_data/HH/` — see [`../../exp_data/HH/README.md`](../../exp_data/HH/README.md).
