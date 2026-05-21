# Phase 3 TODO

## 1. Choose Target System

Decision owner: Phase 3 lead

Open candidates:

- Coupled nonlinear oscillator
- Double pendulum
- Kepler / two-body problem
- Another Hamiltonian system proposed by the Phase 3 lead

Selection criteria:

- Clear Hamiltonian formulation
- Energy-conserving dynamics
- Nontrivial long-time rollout behavior
- Feasible implementation within the project timeline
- Good contrast with Hénon-Heiles

## 2. Derive System Equations

For the chosen system, write:

- State variables
- Hamiltonian H(q, p)
- Hamilton equations
- Initial conditions
- Energy regime or stability regime being tested

Use `system_notes.md` for this.

## 3. Implement Ground Truth

Create or adapt:

- ODE right-hand side
- SciPy solver wrapper
- Energy function
- Optional symplectic integrator comparison

Reference files:

- `../Phase2/HHsystem/utils_HHsystem.py`
- `../Phase1/NLoscillator/utils_NLoscillator.py`

## 4. Implement Neural Solver

Adapt the Phase 2 HH script:

- Change output state dimension if needed
- Replace Hamiltonian-equation residuals
- Replace energy penalty
- Reuse activation classes where appropriate
- Keep the plotting structure close to Phase 2 for easier comparison

Reference file:

- `../Phase2/HHsystem/HamiltonianNet_HenonHeiles.py`

## 5. Run Focused Activation Comparison

Recommended first pass:

- `mySin`
- `AdaptiveSin`
- One custom Phase 2 activation: `GaborActivation` or `DualAdaptiveSin`

Record for each run:

- Activation
- Initial condition
- Training horizon
- Test horizon
- Epochs
- Learning rate
- Hidden width
- Final qualitative result

## 6. Prepare Final Comparison

After Phase 3 results exist, create a compact comparison table with:

- System
- Activation
- Final loss
- Maximum state error
- Mean state error
- Maximum energy drift
- Qualitative outcome

Do this after Phase 3 is implemented so the metrics schema reflects both HH and the new system.
