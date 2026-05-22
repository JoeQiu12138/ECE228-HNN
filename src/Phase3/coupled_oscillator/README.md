# Coupled oscillator — implementation entry point

Physics, goals, checklist, and results interpretation: **[`../README.md`](../README.md)** (single Phase 3 doc).

## Files here

| File | Role |
| --- | --- |
| `p3_train_eval.py` | Train / eval / export CLI |
| `p3_model.py` | Network + Hamiltonian loss |
| `physics_utils.py` | Ground truth |
| `models/` | `model_coupled_<activation>_*.zip` |
| `results/` | `p3_activation_comparison.*`, `phase2_vs_phase3_comparison.*` |

## Commands

```bash
cd src/Phase3/coupled_oscillator
python3 p3_train_eval.py --device cuda --train all
python3 p3_train_eval.py --eval-only --device auto
```

Artifacts: `../../../exp_data/coupled_oscillator/<activation>/`
