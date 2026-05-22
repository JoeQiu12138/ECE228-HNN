# Phase 3 — Cross-domain validation (coupled nonlinear oscillators)

## Goal

Check whether Phase 2 insights on **Hénon–Heiles** transfer to a **second** Hamiltonian system:

```text
Phase 2 HH results  vs.  Phase 3 coupled-oscillator results
```

Main comparison table: [`coupled_oscillator/results/phase2_vs_phase3_comparison.md`](coupled_oscillator/results/phase2_vs_phase3_comparison.md)

---

## Status (implementation checklist)

| Step | Item | Status |
| --- | --- | --- |
| 1 | Choose target system | **Done** — coupled nonlinear oscillators |
| 2 | Hamiltonian + ICs documented | **Done** — see below |
| 3 | Ground truth (`physics_utils.py`) | **Done** |
| 4 | Neural solver (`p3_model.py`, `p3_train_eval.py`) | **Done** |
| 5 | Train/eval 3 activations | **Done** — mySin, AdaptiveSin, GaborActivation |
| 6 | Phase 2 vs Phase 3 table | **Done** |

### Result note (for the report)

On **HH**, Gabor long-time is strongest. On **this coupled system**, **mySin** has the lowest trajectory error; AdaptiveSin and Gabor are worse. That is still a valid Phase 3 outcome: document **where transfer holds and where it does not**, rather than expecting identical ranking.

Latest Phase 3 metrics: [`coupled_oscillator/results/p3_activation_comparison.md`](coupled_oscillator/results/p3_activation_comparison.md)

---

## Physical system

**State:** \([x_1, x_2, p_1, p_2]\), parameters in `X0 = [t_0, x_1, x_2, p_1, p_2, \beta]`.

**Hamiltonian:**

\[
H = \frac{p_1^2 + p_2^2 + x_1^2 + x_2^2}{2} + \beta\, x_1^2 x_2^2
\]

**Equations:**

```text
dx1/dt = p1
dx2/dt = p2
dp1/dt = -x1 - 2*beta*x1*x2^2
dp2/dt = -x2 - 2*beta*x2*x1^2
```

**Initial conditions** (\(\beta = 0.5\)): \(x_1=0.5,\ x_2=-0.3,\ p_1=0.25,\ p_2=0.15\), \(E_0 \approx 0.403\).

**Contrast with HH:** no \(y^3\) term, no escape energy \(1/6\); smooth coupling potential instead of HH’s cubic structure.

---

## Quick start

```bash
cd src/Phase3/coupled_oscillator

python3 p3_train_eval.py --smoke
python3 p3_train_eval.py --device cuda --train all
python3 p3_train_eval.py --export-all --device cuda
python3 p3_train_eval.py --eval-only --device auto
```

`--train` targets: `mysin-long`, `adaptivesin-long`, `gabor-long`, or `all`.

---

## Repository layout

| Path | Role |
| --- | --- |
| [`coupled_oscillator/p3_train_eval.py`](coupled_oscillator/p3_train_eval.py) | CLI (train / eval / export) |
| [`coupled_oscillator/p3_model.py`](coupled_oscillator/p3_model.py) | Network + loss (activations from Phase 2) |
| [`coupled_oscillator/physics_utils.py`](coupled_oscillator/physics_utils.py) | `odeint`, energy, symplectic Euler |
| [`coupled_oscillator/models/`](coupled_oscillator/models/) | Checkpoints |
| [`coupled_oscillator/results/`](coupled_oscillator/results/) | Phase 3 + cross-system tables |
| [`../exp_data/coupled_oscillator/`](../exp_data/coupled_oscillator/) | PNG + error txt |

Phase 2 reference: [`../Phase2/HHsystem/`](../Phase2/HHsystem/)

---

## Experiment protocol

| Item | Value |
| --- | --- |
| Short / long horizon | \(6\pi\) / \(12\pi\) |
| Training points | 500 |
| Hidden width | 80 |
| Short / long epochs | 20k / 50k (`--quick` → 2k / 5k) |
| Activations | mySin, AdaptiveSin, GaborActivation only |

---

## Report guardrails

- Compare **Phase 2 vs Phase 3**, not Phase 1 vs Phase 2.
- Do not rerun every Phase 2 activation here; three representatives are enough.
- Describe the solver as Hamiltonian-equation-informed (physics residual + energy penalty), consistent with Phases 1–2.

---

## Reuse from earlier phases

- **Phase 1** ([`../Phase1/NLoscillator/`](../Phase1/NLoscillator/)): 1D template for loss and plotting.
- **Phase 2** ([`../Phase2/HHsystem/`](../Phase2/HHsystem/)): 4D structure, activation classes, training schedule.

Code details: [`coupled_oscillator/README.md`](coupled_oscillator/README.md)
