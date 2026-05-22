# Phase 3: Cross-Domain Validation

Phase 3 tests whether the architectural and physics-constraint insights from the Hénon-Heiles experiments transfer to a second Hamiltonian system.

The goal is not to redo every Phase 2 ablation. The goal is to choose one additional Hamiltonian system, run a focused set of representative activations, and compare whether the same qualitative behavior appears outside the HH benchmark.

## Current Status

- Target system: TBD
- Hamiltonian and equations of motion: TBD
- Ground-truth solver: TBD
- Activations to compare: TBD
- Metrics and final comparison table: pending Phase 3 results

## Recommended Phase 3 Question

Do the activation priors and energy-conservation constraints that worked on the Hénon-Heiles system also improve long-time stability on another Hamiltonian dynamical system?

## Suggested Minimal Experiment

Pick one new Hamiltonian system and compare:

- `mySin`: periodic baseline
- `AdaptiveSin`: simple learnable-frequency periodic prior
- One Phase 2 custom activation, such as `GaborActivation` or `DualAdaptiveSin`

The final comparison should be Phase 2 vs Phase 3, not Phase 1 vs Phase 2.

## Expected Outputs

- A new system implementation with Hamiltonian equations and energy function
- Ground-truth trajectory from a numerical solver
- Neural trajectory predictions for the selected activations
- Loss, trajectory, and energy plots
- A final comparison table across HH and the Phase 3 system

## Reusable Project Structure

- Phase 1 reference: `../Phase1/NLoscillator/`
- Phase 2 reference: `../Phase2/HHsystem/`
- Existing experiment artifacts: `(../../exp_data/)`
