# HH supplementary studies

Energy tiers + Poincaré at $x=0$. These are **supplementary sensitivity studies**, not replacements for the canonical HH comparison in `hh_train_eval.py`. Checkpoints: `studies/models/`.

**Table:** [`results/energy_tier_comparison.md`](results/energy_tier_comparison.md)  
**Plots:** [`../../../../exp_data/HH_studies/`](../../../../exp_data/HH_studies/)  
**Canonical HH:** [`../results/hh_activation_comparison.md`](../results/hh_activation_comparison.md)

## Tiers (fixed seeds 301 / 302 / 303)

| Tier | $E$ |
| --- | ---: |
| `low_regular` | 0.08 |
| `medium_baseline` | 0.12825 |
| `high_chaotic` | 0.155 (near $1/6$) |

`medium_baseline` matches the canonical HH energy scale, but it uses a tier-generated initial condition and study checkpoints. Interpret it as a sensitivity run, not as a reproduction of the canonical HH row.

## Interpretation

- Energy tiers test whether activation rankings change with energy regime / initial condition.
- Poincaré sections are qualitative phase-space diagnostics, useful for visualizing whether the learned dynamics preserve Hamiltonian structure.
- Report these studies as supplementary evidence. The canonical HH metrics should remain the main Phase 2 table.

## Commands

```bash
cd src/Phase2/HHsystem/studies
python3 energy_tier_study.py --eval-only --device auto --tier all
python3 poincare_section.py --tier high_chaotic --activations GaborActivation mySin AdaptiveSin DualAdaptiveSin --device cuda --n-dense 12000
```

Full retrain: `python3 energy_tier_study.py --device cuda --train all --tier all` (hours).
