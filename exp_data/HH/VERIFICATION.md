# Phase 2 `exp_data/HH/` verification (2026-05-22)

## Figure folders (complete)

| Activation | Path | trajectories | error | loss |
| --- | --- | :---: | :---: | :---: |
| mySin | `baseline/` | yes | yes | yes |
| AdaptiveSin | `adaptivesin/a=1 longtime/` | yes | yes | yes |
| AdaptiveSin (short) | `adaptivesin/short/` | pending long-train export | — | — |
| DualAdaptiveSin | `dualSin/longtime/` | yes | yes | yes |
| DualAdaptiveSin (short) | `dualSin/short/` | see `dualSin/longtime/` or re-run short checkpoint | — | — |
| GaborActivation | `Gabor/short/`, `Gabor/longtime/` | yes | yes | yes |
| LearnablePolynomial | `LearnablePolynomial/` | N/A (failed) | N/A | probe only |

## LearnablePolynomial

- **Status:** FAILED / unstable (short-time probe, 3000 epochs).
- **Artifacts:** `LearnablePolynomial/README.md`, `HenonHeiles_loss_probe.png`.
- **Final probe loss:** ~3.0×10⁵ (diverged); not included in long-time comparison table.

## Quantitative table

Canonical metrics: `src/Phase2/HHsystem/results/hh_activation_comparison.md`

- **GaborActivation (long):** metrics from `exp_data/HH/Gabor/longtime/*.txt` (synced from `exp_data/HH/data/`; max traj. err ≈ 1.59×10⁻³).
- **Other runs:** from checkpoint evaluation at the matching horizon unless `dx.txt` exists in the experiment folder.
- **Regenerate:** `cd src/Phase2/HHsystem && python3 hh_train_eval.py --eval-only`
- **Retrain all runs:** `python3 hh_train_eval.py --train all`
- **Historical Gabor row:** `python3 hh_train_eval.py --eval-only --golden-gabor`
- **Progress:** training prints every 500 epochs (loss, %, ETA); install `tqdm` for a bar. Use `--quick` for 2k+5k epochs smoke test.
