# ECE 228 — Physics-informed Hamiltonian neural solvers

**UC San Diego — final project.** Equation residuals + energy penalties on Hamiltonian systems (HNN-inspired **neural solution ansatz**).

Primary: **Hénon–Heiles**. Also: energy-tier HH studies + **coupled oscillators** (Phase 3).

---

## Layout

```
src/Phase1/NLoscillator/       → exp_data/1D/
src/Phase2/HHsystem/           → exp_data/HH/  (+ studies/ → HH_studies/)
src/Phase3/coupled_oscillator/ → exp_data/coupled_oscillator/
reference/                     → PDFs (incl. ece228_proposal.pdf)
```

---

## Results (cite these tables)

| Phase | Table |
| --- | --- |
| 2 HH | [`src/Phase2/HHsystem/results/hh_activation_comparison.md`](src/Phase2/HHsystem/results/hh_activation_comparison.md) |
| 2 supp. | [`src/Phase2/HHsystem/studies/results/energy_tier_comparison.md`](src/Phase2/HHsystem/studies/results/energy_tier_comparison.md) |
| 3 | [`src/Phase3/coupled_oscillator/results/phase2_vs_phase3_comparison.md`](src/Phase3/coupled_oscillator/results/phase2_vs_phase3_comparison.md) |

**Headline:** HH canonical — Gabor max err **1.59e-3**; coupled system — mySin **6.95e-4**. Rankings are system-dependent.

---

## Quick run

```bash
cd src/Phase2/HHsystem && python3 hh_train_eval.py --eval-only --device auto
cd src/Phase2/HHsystem/studies && python3 energy_tier_study.py --eval-only --device auto --tier all
cd src/Phase3/coupled_oscillator && python3 p3_train_eval.py --eval-only --device auto
```

Phase 1: `cd src/Phase1/NLoscillator` → edit `actF` → `python3 HNN_NLoscillator.py`.

---

## Docs (only these)

| File | Content |
| --- | --- |
| This README | Overview |
| [`exp_data/README.md`](exp_data/README.md) | Artifact folders |
| [`src/Phase2/HHsystem/README.md`](src/Phase2/HHsystem/README.md) | HH CLI + flags |
| [`src/Phase2/HHsystem/studies/README.md`](src/Phase2/HHsystem/studies/README.md) | Energy tiers / Poincaré |
| [`src/Phase3/README.md`](src/Phase3/README.md) | Coupled oscillator |
| [`exp_data/HH/LearnablePolynomial/README.md`](exp_data/HH/LearnablePolynomial/README.md) | Polynomial failure notes |
| [`reference/README.md`](reference/README.md) | PDF index |

---

*Greydanus et al. (HNN); Mattheakis et al. (Phys. Rev. E, 2022).*
