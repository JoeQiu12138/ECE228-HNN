# exp_data — experiment outputs

Figures and `.txt` only. Checkpoints under `src/`. Metrics tables:

| Phase | Table |
| --- | --- |
| 2 HH | [`../src/Phase2/HHsystem/results/hh_activation_comparison.md`](../src/Phase2/HHsystem/results/hh_activation_comparison.md) |
| 2 HH tiers | [`../src/Phase2/HHsystem/studies/results/energy_tier_comparison.md`](../src/Phase2/HHsystem/studies/results/energy_tier_comparison.md) |
| 3 | [`../src/Phase3/coupled_oscillator/results/phase2_vs_phase3_comparison.md`](../src/Phase3/coupled_oscillator/results/phase2_vs_phase3_comparison.md) |

## `1D/` (Phase 1)

`baseline/`, `adaptiveSin/`, `Snake/`, `GELU/`, `tanh/` — PNG per activation. Raw numerics: `1D/data/`. Run: `src/Phase1/NLoscillator/HNN_NLoscillator.py` (edit `actF`).

## `HH/` (Phase 2 canonical)

| Activation | Folder |
| --- | --- |
| mySin | `baseline/` |
| AdaptiveSin | `adaptivesin/a=1 longtime/` |
| DualAdaptiveSin | `dualSin/longtime/` |
| GaborActivation | `Gabor/longtime/` |
| LearnablePolynomial | `LearnablePolynomial/` (failed probe — see folder README) |

Archive txt: `HH/data/` (optional `--golden-gabor` in `hh_train_eval.py`).

## `HH_studies/` (Phase 2 supplementary)

`energy_tier/<tier>/<activation>/`, `poincare/<tier>/<activation>/`. Docs: [`../src/Phase2/HHsystem/studies/README.md`](../src/Phase2/HHsystem/studies/README.md).

## `coupled_oscillator/` (Phase 3)

`baseline/` (mySin), `adaptiveSin/`, `Gabor/`.
