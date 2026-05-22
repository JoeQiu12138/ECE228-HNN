# Phase 3 — Coupled nonlinear oscillators

**Question:** Do Phase 2 HH activation trends transfer to another Hamiltonian system?

**Table:** [`coupled_oscillator/results/phase2_vs_phase3_comparison.md`](coupled_oscillator/results/phase2_vs_phase3_comparison.md)  
**Artifacts:** [`../../exp_data/coupled_oscillator/`](../../exp_data/coupled_oscillator/)

**Outcome:** In the canonical HH run, Gabor gives the best rollout among the tested activations. In the coupled oscillator run, mySin is better.

---

## System

$H = (p_1^2+p_2^2+x_1^2+x_2^2)/2 + \beta x_1^2 x_2^2$, $\beta=0.5$. State $[x_1,x_2,p_1,p_2]$.

---

## Run

```bash
cd src/Phase3/coupled_oscillator
python3 p3_train_eval.py --smoke
python3 p3_train_eval.py --device cuda --train all
python3 p3_train_eval.py --eval-only --device auto
```

| File | Role |
| --- | --- |
| `p3_train_eval.py` | CLI |
| `p3_model.py` | Network + loss |
| `physics_utils.py` | Ground truth |
| `models/` | Checkpoints |
| `results/` | Phase 3 + cross-system tables |

Activations: mySin, AdaptiveSin, GaborActivation (same classes as Phase 2).

Phase 2 reference: [`../Phase2/HHsystem/`](../Phase2/HHsystem/)
