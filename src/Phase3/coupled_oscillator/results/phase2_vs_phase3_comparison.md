# Phase 2 (HH) vs Phase 3 (coupled oscillators)

Long-time rows only; activations: mySin, AdaptiveSin, GaborActivation.

| System | Activation | Final loss | Max traj. err | Mean traj. err | Max E drift | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Hénon-Heiles | mySin | 3.166e-05 | 9.230e-01 | 4.964e-01 | 5.491e-03 | visible drift or phase error |
| Hénon-Heiles | AdaptiveSin | 1.923e-02 | 6.821e-01 | 5.083e-01 | 1.282e-01 | visible drift or phase error |
| Hénon-Heiles | GaborActivation | 8.792e-04 | 1.586e-03 | 7.585e-04 | 6.022e-04 | strong match to ground truth |
| Coupled nonlinear oscillators | mySin | 1.450e-07 | 6.954e-04 | 3.428e-04 | 1.424e-04 | strong match to ground truth |
| Coupled nonlinear oscillators | AdaptiveSin | 3.766e-03 | 1.165e+00 | 7.526e-01 | 8.045e-02 | visible drift or phase error |
| Coupled nonlinear oscillators | GaborActivation | 1.598e-02 | 7.192e-01 | 4.455e-01 | 2.237e-01 | visible drift or phase error |
