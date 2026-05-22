# Phase 1 — 1D nonlinear oscillator

**Code and checkpoints:** [`NLoscillator/`](NLoscillator/)  
**All experiment artifacts:** [`../exp_data/1D/`](../exp_data/1D/)

| Path | Role |
| --- | --- |
| `NLoscillator/HNN_NLoscillator.py` | Main script — edit `actF`, then run |
| `NLoscillator/utils_NLoscillator.py` | Ground truth, energy, symplectic Euler |
| `NLoscillator/models/` | Checkpoints (e.g. `model_NL.zip`) |
| `../exp_data/1D/<activation>/` | PNG plots per activation |
| `../exp_data/1D/data/` | Historical `.txt` archive (moved from `NLoscillator/data/`) |

## How to run

```bash
cd src/Phase1/NLoscillator
# Uncomment the experiment block you want in HNN_NLoscillator.py, set actF, then:
python3 HNN_NLoscillator.py
```

Outputs (aligned with Phase 2):

- **Figures:** `exp_data/1D/<activation>/nonlinearOscillator_*.png` (not the `NLoscillator/` code folder)
- **Loss archive:** `exp_data/1D/data/loss.txt`
- **Checkpoint:** `NLoscillator/models/model_NL.zip`

Activation → folder mapping: `mySin` → `baseline/`, `AdaptiveSin` → `adaptiveSin/`, `Snake` → `Snake/`, `GELU` → `GELU/`, `Tanh` → `tanh/`.

More detail: [`../exp_data/1D/README.md`](../exp_data/1D/README.md).
