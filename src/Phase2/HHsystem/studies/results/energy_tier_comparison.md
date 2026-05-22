# Optional study: HH energy tiers

Escape energy $E_{esc}=1/6\approx0.1667$. Not part of main `hh_train_eval.py` table.

`medium_baseline` matches the canonical HH energy scale, but uses a tier-generated initial condition and study checkpoints. Interpret this table as sensitivity analysis, not as a reproduction of the canonical HH row.

| Tier | E_target | E0 | Activation | Final loss | Max traj err | Max E drift | Qualitative |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| low_regular | 0.0800 | 0.0800 | mySin | 1.112e-06 | 1.767e-03 | 1.439e-04 | strong |
| low_regular | 0.0800 | 0.0800 | AdaptiveSin | 1.015e-02 | 4.727e-01 | 7.990e-02 | poor / divergent |
| low_regular | 0.0800 | 0.0800 | DualAdaptiveSin | 2.571e-05 | 7.917e-01 | 2.069e-03 | poor / divergent |
| low_regular | 0.0800 | 0.0800 | GaborActivation | 2.496e-03 | 4.646e-01 | 7.999e-02 | poor / divergent |
| medium_baseline | 0.1283 | 0.1282 | mySin | 1.536e-05 | 9.743e-01 | 2.449e-03 | poor / divergent |
| medium_baseline | 0.1283 | 0.1282 | AdaptiveSin | 1.317e-02 | 1.464e+00 | 3.887e-02 | poor / divergent |
| medium_baseline | 0.1283 | 0.1282 | DualAdaptiveSin | 5.178e-05 | 9.271e-01 | 4.362e-03 | poor / divergent |
| medium_baseline | 0.1283 | 0.1282 | GaborActivation | 2.686e-03 | 1.442e+00 | 7.944e-02 | poor / divergent |
| high_chaotic | 0.1550 | 0.1550 | mySin | 2.395e-03 | 1.560e+00 | 4.552e-02 | poor / divergent |
| high_chaotic | 0.1550 | 0.1550 | AdaptiveSin | 1.748e-03 | 1.119e+00 | 4.698e-02 | poor / divergent |
| high_chaotic | 0.1550 | 0.1550 | DualAdaptiveSin | 1.137e-03 | 6.467e-01 | 4.239e-02 | poor / divergent |
| high_chaotic | 0.1550 | 0.1550 | GaborActivation | 1.509e-04 | 1.499e+00 | 1.588e-02 | poor / divergent |
