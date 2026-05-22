# `exp_data/1D/data/` — Phase 1 numerics archive

Moved from `src/Phase1/NLoscillator/data/`; **files were not deleted**, only relocated.

## Purpose

| Content | Notes |
| --- | --- |
| Root `dx.txt`, `x.txt`, `loss.txt`, … | Early 1D oscillator canonical numerics |
| `Euler100/` | Symplectic Euler comparison at 100 points; optional for the main report |

**For the report:** use figures under sibling folders (`baseline/`, `adaptiveSin/`, …). This directory is a **backup archive** for raw `.txt`.

## Main root files

| File | Meaning |
| --- | --- |
| `x.txt`, `px.txt` | Network state |
| `dx.txt`, `dp.txt` | Error vs ground truth |
| `E.txt`, `loss.txt`, `t.txt` | Energy, training loss, time |

## Related commands

```bash
cd src/Phase1/NLoscillator
python3 HNN_NLoscillator.py
```

Loss is written to `loss.txt` here; PNGs go to `exp_data/1D/<activation>/`.
