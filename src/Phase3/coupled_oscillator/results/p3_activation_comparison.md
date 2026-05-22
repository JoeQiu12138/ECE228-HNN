# Phase 3 — coupled oscillator activation comparison

System: two coupled nonlinear oscillators, $H = (p_1^2+p_2^2+x_1^2+x_2^2)/2 + \beta x_1^2 x_2^2$, $\beta=0.5$.
Long-time: $t \in [0, 12\pi]$, 500 points, warm-start from $6\pi$.

| Activation | Final loss | Max traj. err | Mean traj. err | Max E drift | Mean E drift | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mySin | 1.450e-07 | 6.954e-04 | 3.428e-04 | 1.424e-04 | 5.496e-05 | strong match to ground truth |
| AdaptiveSin | 3.766e-03 | 1.165e+00 | 7.526e-01 | 8.045e-02 | 2.806e-02 | visible drift or phase error |
| GaborActivation | 1.598e-02 | 7.192e-01 | 4.455e-01 | 2.237e-01 | 1.343e-01 | visible drift or phase error |
