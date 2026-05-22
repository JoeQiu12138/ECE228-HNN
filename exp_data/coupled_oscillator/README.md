# exp_data/coupled_oscillator — Phase 3 artifacts

| Folder | Activation |
| --- | --- |
| `baseline/` | mySin |
| `adaptiveSin/` | AdaptiveSin |
| `Gabor/` | GaborActivation |

Regenerate:

```bash
cd src/Phase3/coupled_oscillator
python3 p3_train_eval.py --device cuda --train all
python3 p3_train_eval.py --eval-only --device auto
```

Code: [`src/Phase3/coupled_oscillator/README.md`](../../src/Phase3/coupled_oscillator/README.md)
