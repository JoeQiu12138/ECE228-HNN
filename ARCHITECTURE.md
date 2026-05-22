# Repository layout

```
ECE228-HNN/
├── src/                    # all source code
│   ├── Phase1/NLoscillator/
│   ├── Phase2/HHsystem/
│   └── Phase3/coupled_oscillator/
├── exp_data/               # experiment outputs (figures, txt, archives)
│   ├── 1D/                 # Phase 1
│   ├── HH/                 # Phase 2
│   └── coupled_oscillator/ # Phase 3
└── reference/              # PDF papers and lecture notes
```

## Roles

| Tree | Holds |
| --- | --- |
| **src** | Python scripts, checkpoints under `src/Phase2/HHsystem/models/`, comparison tables under `results/` |
| **exp_data** | PNG plots, `dx.txt`, archives under `exp_data/HH/data/` — not source code |
| **reference** | `HNN.pdf`, `tradition.pdf`, course lecture PDF |

## Phase 2 CLI (`src/Phase2/HHsystem/`)

```bash
cd src/Phase2/HHsystem
python3 hh_train_eval.py --eval-only --device auto
python3 hh_train_eval.py --device cuda --train all
python3 hh_train_eval.py --eval-only --golden-gabor   # optional Gabor archive row
```

| Flag | Meaning |
| --- | --- |
| `--eval-only` | Score checkpoints; metrics from `exp_data/HH/<run>/` exports |
| `--train TARGET …` / `--train all` | Retrain selected runs |
| `--golden-gabor` | Sync `exp_data/HH/data/` → `Gabor/longtime/`, score archive for Gabor long |

Default eval does **not** sync the Gabor archive (trust your checkpoint exports).

## Phase 1

```bash
cd src/Phase1/NLoscillator
python3 HNN_NLoscillator.py
```

Figures → `exp_data/1D/<activation>/`; numerics → `exp_data/1D/data/`.

## Phase 3

```bash
cd src/Phase3/coupled_oscillator
python3 p3_train_eval.py --device cuda --train all
python3 p3_train_eval.py --eval-only --device auto
```

Cross-system table: `src/Phase3/coupled_oscillator/results/phase2_vs_phase3_comparison.md`
