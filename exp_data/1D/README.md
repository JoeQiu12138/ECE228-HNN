# exp_data/1D — Phase 1 artifact directory

PNG plots, per-activation folders, and the numerics archive for the **1D nonlinear oscillator**.

## Layout

```
exp_data/1D/
├── data/              # moved from src/Phase1/NLoscillator/data/ (do not delete)
├── baseline/          # mySin
├── adaptiveSin/       # AdaptiveSin
├── Snake/
├── GELU/
└── tanh/              # Tanh
```

## How to run

```bash
cd src/Phase1/NLoscillator
python3 HNN_NLoscillator.py
```

1. Set `actF` in `odeNet_NLosc_MM`.
2. Run the script — figures are saved under the matching folder above.
3. Training loss history is appended to `data/loss.txt` when `trainModel` runs.

## Artifact map

| Activation | Folder | Checkpoint |
| --- | --- | --- |
| mySin | `baseline/` | `src/Phase1/NLoscillator/models/model_NL.zip` |
| AdaptiveSin | `adaptiveSin/` | same (change `actF` and re-train) |
| Snake | `Snake/` | same |
| GELU | `GELU/` | same |
| Tanh | `tanh/` | same |

Numerics archive: [`data/README.md`](data/README.md)  
Phase 1 code overview: [`../../src/Phase1/README.md`](../../src/Phase1/README.md)
