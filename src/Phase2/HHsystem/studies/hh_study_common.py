"""Shared helpers for optional HH studies (not used by hh_train_eval.py)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

STUDIES_ROOT = Path(__file__).resolve().parent
HH_ROOT = STUDIES_ROOT.parent
REPO_ROOT = STUDIES_ROOT.parents[3]  # studies -> HHsystem -> Phase2 -> src -> repo root
EXP_STUDIES = REPO_ROOT / "exp_data" / "HH_studies"
MODELS_STUDIES = STUDIES_ROOT / "models"
RESULTS_STUDIES = STUDIES_ROOT / "results"
MAIN_MODELS = HH_ROOT / "models"

LAM = 1.0
ESCAPE_E = 1.0 / 6.0
CANONICAL_X0 = [0.0, 0.3, -0.3, 0.3, 0.15, LAM]
N_TRAIN = 500
SHORT_T = 6 * np.pi
LONG_T = 12 * np.pi

ENERGY_TIERS = {
    "low_regular": 0.08,
    "medium_baseline": 0.12825,
    "high_chaotic": 0.155,
}

# Fixed seeds per tier (reproducible across machines; do not use hash(name)).
TIER_IC_SEEDS = {
    "low_regular": 301,
    "medium_baseline": 302,
    "high_chaotic": 303,
}

ACTIVATIONS_ALL = (
    "mySin",
    "AdaptiveSin",
    "DualAdaptiveSin",
    "GaborActivation",
    "LearnablePolynomial",
)


def hh_energy_scalar(x0: float, y0: float, px0: float, py0: float, lam: float = LAM) -> float:
    return float(
        0.5 * (px0**2 + py0**2)
        + 0.5 * (x0**2 + y0**2)
        + lam * (x0**2 * y0 - y0**3 / 3.0)
    )


def find_initial_state(
    E_target: float,
    lam: float = LAM,
    seed: int = 228,
    n_samples: int = 8000,
) -> Tuple[List[float], float]:
    """Search random ICs near the canonical point for target energy."""
    rng = np.random.default_rng(seed)
    bx, by, bpx, bpy = 0.3, -0.3, 0.3, 0.15
    best = None
    best_err = float("inf")
    for _ in range(n_samples):
        sc = rng.uniform(0.35, 1.45, size=4)
        x0 = bx * sc[0]
        y0 = by * sc[1]
        px0 = bpx * sc[2]
        py0 = bpy * sc[3]
        E = hh_energy_scalar(x0, y0, px0, py0, lam)
        err = abs(E - E_target)
        if err < best_err:
            best_err = err
            best = (x0, y0, px0, py0)
    assert best is not None
    x0, y0, px0, py0 = best
    return [0.0, x0, y0, px0, py0, lam], hh_energy_scalar(x0, y0, px0, py0, lam)


def tier_initial_state(tier: str) -> Tuple[List[float], float]:
    if tier not in ENERGY_TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Choose: {', '.join(ENERGY_TIERS)}")
    Et = ENERGY_TIERS[tier]
    X0, E = find_initial_state(Et, seed=TIER_IC_SEEDS[tier])
    return X0, E


def tier_X0_table() -> dict[str, list[float]]:
    out = {}
    for name in ENERGY_TIERS:
        X0, E = tier_initial_state(name)
        out[name] = X0
        print(f"[tier] {name}: target E={ENERGY_TIERS[name]:.5f}, found E={E:.5f}, X0={X0[1:]}", flush=True)
    return out


def activation_slug(name: str) -> str:
    return {
        "mySin": "mysin",
        "AdaptiveSin": "adaptivesin",
        "DualAdaptiveSin": "dualsin",
        "GaborActivation": "gabor",
        "LearnablePolynomial": "polynomial",
    }[name]


def study_checkpoint(tier: str, activation: str) -> Path:
    return MODELS_STUDIES / f"model_HH_{tier}_{activation_slug(activation)}_longtime.zip"


def study_experiment_dir(tier: str, activation: str) -> Path:
    return EXP_STUDIES / "energy_tier" / tier / activation_slug(activation)
