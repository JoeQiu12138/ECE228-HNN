# Optional HH studies (not in main proposal)

Scripts here are **separate** from [`../hh_train_eval.py`](../hh_train_eval.py).  
Use them as supplementary / report add-ons.

| Script | Role | Outputs |
| --- | --- | --- |
| [`energy_tier_study.py`](energy_tier_study.py) | Train/eval low / medium / high energy | `studies/models/`, `exp_data/HH_studies/energy_tier/`, `studies/results/energy_tier_comparison.md` |
| [`poincare_section.py`](poincare_section.py) | Poincaré section at $x=0$ | `exp_data/HH_studies/poincare/<tier>/` |

Shared helpers: [`hh_study_common.py`](hh_study_common.py) — tier IC seeds are **fixed** (`301/302/303`), not `hash()`.

## Energy tiers

| Tier | Target $E$ | IC seed |
| --- | ---: | ---: |
| `low_regular` | 0.08 | 301 |
| `medium_baseline` | 0.12825 | 302 |
| `high_chaotic` | 0.155 | 303 |

`LearnablePolynomial` is skipped for tier training (unstable on HH).

## Poincaré `--tier`

| `--tier` | Initial condition | Checkpoints |
| --- | --- | --- |
| `canonical` (default) | Main HH run (`E_0\approx 0.128`) | `../models/model_HH_*_longtime.zip` |
| `low_regular` / `medium_baseline` / `high_chaotic` | Fixed tier IC from `hh_study_common` | `studies/models/model_HH_<tier>_*_longtime.zip` |

Run tier training **before** `poincare_section.py --tier high_chaotic`.

## Report wording

- Label as **supplementary analysis**.
- Chaotic / high-$E$: report trajectory error + energy drift; Poincaré is qualitative only.
