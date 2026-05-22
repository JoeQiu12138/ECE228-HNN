# Phase 2 — 2D Hénon–Heiles (HH) system

Phase 2 **code and checkpoints** live under [`HHsystem/`](HHsystem/).  
**Figures, error `.txt` files, and the historical Gabor archive** live under [`../exp_data/HH/`](../exp_data/HH/).

| Path | Contents |
| --- | --- |
| [`HHsystem/`](HHsystem/) | Scripts (`hh_train_eval.py`, …), `models/*.zip`, `results/` comparison table |
| [`../exp_data/HH/`](../exp_data/HH/) | PNG plots, trajectory/error txt, `data/` archive |

## Where to start

```bash
cd src/Phase2/HHsystem
python3 hh_train_eval.py --smoke          # CPU + CUDA sanity check
python3 hh_train_eval.py --eval-only --device auto   # refresh comparison table
```

Full CLI reference, per-script roles, and CPU/GPU workflow:  
[`HHsystem/README.md`](HHsystem/README.md) and [`HHsystem/WORKFLOW.md`](HHsystem/WORKFLOW.md).
