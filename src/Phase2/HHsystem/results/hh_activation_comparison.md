# Phase 2 HH activation comparison

Protocol: Hénon–Heiles, X0=[0, 0.3, -0.3, 0.3, 0.15, 1], low-energy regime.
Long-time rows: t in [0, 12π], 500 train/test points, 50k epochs (5e-3 lr) after short warm-start where noted.
Trajectory error = L2 norm of (x,y,px,py) vs SciPy `odeint` ground truth. Energy drift = |E_net - E0|.

## Long-time comparison (primary table)

| Activation | Final train loss | Max traj. error | Mean traj. error | Max energy drift | Mean energy drift | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GaborActivation | 8.792e-04 | 1.586e-03 | 7.585e-04 | 6.022e-04 | 1.746e-04 | strong match to ground truth |
| mySin | 3.166e-05 | 9.230e-01 | 4.964e-01 | 5.491e-03 | 1.141e-03 | visible drift or phase error |
| AdaptiveSin | 1.923e-02 | 6.821e-01 | 5.083e-01 | 1.282e-01 | 1.191e-01 | visible drift or phase error |
| DualAdaptiveSin | 1.235e-04 | 9.904e-01 | 6.279e-01 | 1.211e-02 | 1.853e-03 | visible drift or phase error |

## All runs (including short / probe)

| Activation | Horizon | Final train loss | Max traj. error | Mean traj. error | Max E drift | Mean E drift | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mySin | long | 3.166e-05 | 9.230e-01 | 4.964e-01 | 5.491e-03 | 1.141e-03 | visible drift or phase error |
| AdaptiveSin | long | 1.923e-02 | 6.821e-01 | 5.083e-01 | 1.282e-01 | 1.191e-01 | visible drift or phase error |
| DualAdaptiveSin | long | 1.235e-04 | 9.904e-01 | 6.279e-01 | 1.211e-02 | 1.853e-03 | visible drift or phase error |
| GaborActivation | long | 8.792e-04 | 1.586e-03 | 7.585e-04 | 6.022e-04 | 1.746e-04 | strong match to ground truth |
| GaborActivation | short | 5.476e-03 | 6.094e-01 | 3.154e-01 | 1.277e-01 | 6.576e-02 | visible drift or phase error |
| DualAdaptiveSin | short | 4.204e-05 | 5.256e-01 | 1.884e-01 | 4.634e-03 | 1.430e-03 | visible drift or phase error |
| AdaptiveSin | short | 1.923e-02 | 6.701e-01 | 5.005e-01 | 1.282e-01 | 1.228e-01 | visible drift or phase error |
| LearnablePolynomial | probe | 3.046e+05 | N/A | N/A | N/A | N/A | unstable / not used for long-time HH |
