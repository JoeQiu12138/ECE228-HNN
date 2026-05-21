# Phase 3 Handoff

This handoff is meant to help the Phase 3 lead start quickly without locking in the scientific direction too early.

## Phase 3 Goal

Validate whether the Phase 2 conclusions from the Hénon-Heiles system transfer to another Hamiltonian system.

The central comparison should be:

```text
Phase 2 HH benchmark results
vs.
Phase 3 new Hamiltonian-system results
```

## What Is Already Available

Phase 1 provides a simpler conservative baseline:

- 1D nonlinear oscillator training script
- 1D Hamiltonian-equation residual loss
- Energy function
- Ground-truth solver
- Trajectory and error plotting

Path:

```text
../Phase1/NLoscillator/
```

Phase 2 provides the main HH benchmark:

- Hénon-Heiles training script
- 4D state output: x, y, px, py
- Hamiltonian-equation residual loss
- Energy-conservation penalty
- SciPy solver comparison
- Symplectic Euler comparison
- Activation classes: `mySin`, `AdaptiveSin`, `DualAdaptiveSin`, `LearnablePolynomial`, `GaborActivation`
- Trajectory, energy, and error plots

Path:

```text
../Phase2/HHsystem/
```

## Phase 2 Candidates Worth Carrying Into Phase 3

`mySin`

Periodic baseline. Useful as the simplest conservative-dynamics prior.

`AdaptiveSin`

Simple learnable-frequency prior. Good first candidate for transfer because it is interpretable and low-risk.

`DualAdaptiveSin`

Tests the multi-frequency hypothesis from Phase 2. Useful if the new system has coupled or multi-timescale behavior.

`GaborActivation`

Tests localized oscillatory structure. Useful if the lead wants to examine whether local resonance-style priors generalize.

`LearnablePolynomial`

Interesting but riskier. It may be useful as a negative/control experiment if the lead wants to discuss instability.

## Decisions Still Open

- Target Hamiltonian system
- Exact Hamiltonian and state variables
- Initial conditions and energy regime
- Which activations to rerun
- Whether to include a symplectic numerical baseline
- Final metrics schema

## Suggested Implementation Order

1. Pick the Phase 3 system.
2. Write the Hamiltonian and equations in `system_notes.md`.
3. Implement ground truth and energy function.
4. Copy the Phase 2 neural-solver structure and replace the equations.
5. Run `mySin` first as a baseline.
6. Run one or two transfer activations.
7. Only then create the final Phase 2 vs Phase 3 metrics table.

## Guardrails

- Avoid building the final metrics table before Phase 3 results exist.
- Keep Phase 1 as supporting evidence, not the main comparison.
- Phrase the implementation as an HNN-inspired, Hamiltonian-equation-informed neural solver unless a scalar-Hamiltonian network is added.
- Keep activation comparisons focused. Phase 3 should show transfer, not repeat every Phase 2 trial.
