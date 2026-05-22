# `exp_data/HH/data/` — Gabor historical numerics

This tree was moved from `src/Phase2/HHsystem/data/`; **files were not deleted**, only relocated.

## Purpose

| Content | Notes |
| --- | --- |
| Root `dx.txt`, `x.txt`, `loss.txt`, … | Author’s canonical **long-time Gabor** numerics |
| `solver/`, `Euler10/`, `Euler200/` | Side experiments; optional for the main report |

**For the report:** use figures and primary txt under sibling folders (`Gabor/longtime/`, `baseline/`, …). This directory is a **backup archive**.

Use `hh_train_eval.py --eval-only --golden-gabor` to **copy** canonical root files into `../Gabor/longtime/` and score Gabor long from this archive. Normal `--eval-only` does **not** sync here (metrics come from your checkpoint exports).

## Main root files

| File | Meaning |
| --- | --- |
| `x.txt`, `y.txt`, `px.txt`, `py.txt` | Network state |
| `dx.txt` … `dpy.txt` | Error vs ground truth |
| `E.txt`, `loss.txt`, `loss_12.txt`, `t.txt` | Energy, loss, time |

## Related commands

```bash
cd src/Phase2/HHsystem
python3 hh_train_eval.py --eval-only --device auto   # sync + refresh table
```
