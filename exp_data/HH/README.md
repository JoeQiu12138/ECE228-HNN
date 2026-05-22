# exp_data/HH — unified artifact directory

PNG plots, error `.txt` files, and the historical Gabor archive live here.  
**Train, evaluate, and build the comparison table** from code:

```bash
cd src/Phase2/HHsystem
python3 hh_train_eval.py --eval-only --device auto
```

Docs: [`src/Phase2/HHsystem/README.md`](../../src/Phase2/HHsystem/README.md), [`WORKFLOW.md`](../../src/Phase2/HHsystem/WORKFLOW.md).

## Layout

```
exp_data/HH/
├── data/              # moved from src/Phase2/HHsystem/data/ (Gabor archive; do not delete)
├── baseline/          # mySin long
├── adaptivesin/       # AdaptiveSin long / short
├── dualSin/           # DualSin long / short
├── Gabor/             # Gabor long / short
└── LearnablePolynomial/
```

## Artifact map

| Activation | Folder | Stage | Checkpoint (`src/Phase2/HHsystem/models/`) |
| --- | --- | --- | --- |
| mySin | `baseline/` | long | `model_HH_mySin_longtime.zip` |
| AdaptiveSin | `adaptivesin/a=1 longtime/` | long | `model_HH_adaptiveSin_longtime.zip` |
| AdaptiveSin | `adaptivesin/short/` | short | `model_HH_adaptiveSin_shorttime.zip` |
| DualAdaptiveSin | `dualSin/longtime/` | long | `model_HH_dualSin_longtime.zip` |
| DualAdaptiveSin | `dualSin/short/` | short | `model_HH_dualSin_shorttime.zip` |
| GaborActivation | `Gabor/longtime/` | long | `model_HH_Gabor_longtime.zip` |
| GaborActivation | `Gabor/short/` | short | `model_HH_Gabor_short_time.zip` |
| LearnablePolynomial | `LearnablePolynomial/` | probe | see that folder’s README |

## Regenerating artifacts

| Goal | Command (from `src/Phase2/HHsystem/`) |
| --- | --- |
| Refresh metrics table only | `python3 hh_train_eval.py --eval-only --device auto` |
| Export plots/txt from checkpoints | `python3 hh_train_eval.py --export-all --device cuda` |
| Retrain all | `python3 hh_train_eval.py --device cuda --train all` |
| Historical Gabor row | `python3 hh_train_eval.py --eval-only --golden-gabor` |

Comparison table: [`src/Phase2/HHsystem/results/hh_activation_comparison.md`](../../src/Phase2/HHsystem/results/hh_activation_comparison.md)
