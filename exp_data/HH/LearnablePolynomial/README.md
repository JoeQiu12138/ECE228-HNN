# LearnablePolynomial — HH experiment notes

## Status: FAILED / unstable

Phase 2 uses a learnable cubic activation `f(x) = c1*x + c2*x^2 + c3*x^3` intended to mimic HH potential nonlinearity.

## Probe settings
- Horizon: t in [0, 6 pi] (18.8496)
- Training points: 500
- Hidden width: 80
- Epochs: 3000
- Learning rate: 0.008
- Wall time (min): 0.38

## Probe outcome
- Final loss: 3.045796e+05
- Max finite loss: 2.321357e+12
- Epochs logged: 3000
- Diverged (loss > 1e3 or non-finite): **True**

## Artifacts
- `HenonHeiles_loss_probe.png` — probe training curve
- `../../../src/Phase2/HHsystem/models/model_HH_polynomial_probe.zip` — partial checkpoint (if saved before NaN)
