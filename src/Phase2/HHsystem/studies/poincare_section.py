#!/usr/bin/env python3
"""
Optional HH study: Poincaré section at x=0 (y vs py).

Uses main checkpoints when --tier canonical (default).
Uses studies/models/ + tier IC when --tier is low_regular | medium_baseline | high_chaotic.

Usage (from src/Phase2/HHsystem/studies/):
  python3 poincare_section.py --device auto
  python3 poincare_section.py --tier high_chaotic --activations GaborActivation mySin --device cuda
  python3 poincare_section.py --checkpoint ../models/model_HH_Gabor_longtime.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HH_ROOT = Path(__file__).resolve().parent.parent
if str(HH_ROOT) not in sys.path:
    sys.path.insert(0, str(HH_ROOT))

from hh_model import load_model_from_checkpoint, parametricSolutions, resolve_device  # noqa: E402
from hh_physics_utils import HHsolution  # noqa: E402

from hh_study_common import (  # noqa: E402
    CANONICAL_X0,
    ENERGY_TIERS,
    EXP_STUDIES,
    LONG_T,
    MAIN_MODELS,
    N_TRAIN,
    activation_slug,
    hh_energy_scalar,
    study_checkpoint,
    tier_initial_state,
)

NEURONS = 80
POINCARE_TIERS = ("canonical", *tuple(ENERGY_TIERS.keys()))


def poincare_points(t: np.ndarray, x: np.ndarray, y: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collect (y, py) when trajectory crosses x=0 (linear interpolation)."""
    ys: list[float] = []
    pys: list[float] = []
    for i in range(len(t) - 1):
        x1, x2 = x[i], x[i + 1]
        if x1 == 0.0:
            ys.append(y[i])
            pys.append(py[i])
            continue
        if x2 == 0.0:
            ys.append(y[i + 1])
            pys.append(py[i + 1])
            continue
        if x1 * x2 < 0.0:
            alpha = abs(x1) / (abs(x1) + abs(x2))
            ys.append((1 - alpha) * y[i] + alpha * y[i + 1])
            pys.append((1 - alpha) * py[i] + alpha * py[i + 1])
    return np.array(ys), np.array(pys)


def default_checkpoint(activation: str, tier: str = "canonical") -> Path:
    if tier != "canonical":
        return study_checkpoint(tier, activation)
    slug = activation_slug(activation)
    if activation == "GaborActivation":
        return MAIN_MODELS / "model_HH_Gabor_longtime.zip"
    if activation == "mySin":
        return MAIN_MODELS / "model_HH_mySin_longtime.zip"
    if activation == "AdaptiveSin":
        return MAIN_MODELS / "model_HH_adaptiveSin_longtime.zip"
    if activation == "DualAdaptiveSin":
        return MAIN_MODELS / "model_HH_dualSin_longtime.zip"
    if activation == "LearnablePolynomial":
        return MAIN_MODELS / "model_HH_polynomial_longtime.zip"
    return MAIN_MODELS / f"model_HH_{slug}_longtime.zip"


def resolve_X0(tier: str) -> list[float]:
    if tier == "canonical":
        return list(CANONICAL_X0)
    X0, _ = tier_initial_state(tier)
    return X0


def plot_poincare_for_activation(
    activation: str,
    checkpoint: Path,
    out_dir: Path,
    n_dense: int,
    device: str | None,
    *,
    tier: str = "canonical",
    X0: list[float] | None = None,
) -> None:
    if not checkpoint.exists():
        print(f"SKIP (missing ckpt): {checkpoint}", flush=True)
        return

    X0 = X0 if X0 is not None else resolve_X0(tier)
    t0, x0, y0, px0, py0, lam = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    E0 = hh_energy_scalar(x0, y0, px0, py0, lam)
    t_num = np.linspace(t0, LONG_T, n_dense)
    x_gt, y_gt, px_gt, py_gt = HHsolution(n_dense, t_num, x0, y0, px0, py0, lam)

    dev = resolve_device(device)
    model, _ = load_model_from_checkpoint(checkpoint, NEURONS, device=device)
    model.eval()
    t_t = torch.linspace(t0, LONG_T, n_dense, device=dev).reshape(-1, 1)
    t_t.requires_grad = True
    x, y, px, py = parametricSolutions(t_t, model, X0)
    x_nn = x.detach().cpu().numpy()[:, 0]
    y_nn = y.detach().cpu().numpy()[:, 0]
    py_nn = py.detach().cpu().numpy()[:, 0]

    y_p_gt, py_p_gt = poincare_points(t_num, x_gt, y_gt, py_gt)
    y_p_nn, py_p_nn = poincare_points(t_num, x_nn, y_nn, py_nn)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = activation_slug(activation)
    tier_note = "canonical (main HH IC)" if tier == "canonical" else f"tier={tier}, $E_0\\approx{E0:.4f}$"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, yy, ppy, title in [
        (axes[0], y_p_gt, py_p_gt, "Ground truth"),
        (axes[1], y_p_nn, py_p_nn, "Neural network"),
    ]:
        ax.scatter(yy, ppy, s=6, alpha=0.55, c="C0")
        ax.set_xlabel("y")
        ax.set_ylabel(r"$p_y$")
        ax.set_title(f"Poincaré section $x=0$ — {title}")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"{activation} — Hénon–Heiles, {tier_note}, $t\\in[0,{LONG_T/np.pi:.0f}\\pi]$"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_poincare_x0.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(y_p_gt, py_p_gt, s=8, alpha=0.4, c="green", label="truth")
    ax2.scatter(y_p_nn, py_p_nn, s=8, alpha=0.4, c="blue", label="NN")
    ax2.set_xlabel("y")
    ax2.set_ylabel(r"$p_y$")
    ax2.set_title(f"Overlay Poincaré $x=0$ — {activation}")
    ax2.legend()
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{prefix}_poincare_x0_overlay.png", dpi=150)
    plt.close(fig2)

    np.savetxt(out_dir / f"{prefix}_poincare_y.txt", y_p_nn)
    np.savetxt(out_dir / f"{prefix}_poincare_py.txt", py_p_nn)
    print(f"Wrote Poincaré plots -> {out_dir} ({len(y_p_nn)} NN crossings)", flush=True)


def resolve_activations(raw: list[str]) -> list[str]:
    aliases = {
        "mysin": "mySin",
        "adaptivesin": "AdaptiveSin",
        "dualsin": "DualAdaptiveSin",
        "gabor": "GaborActivation",
        "polynomial": "LearnablePolynomial",
    }
    known = {
        "mySin",
        "AdaptiveSin",
        "DualAdaptiveSin",
        "GaborActivation",
        "LearnablePolynomial",
    }
    out = []
    for a in raw:
        key = aliases.get(a.lower(), a)
        if key not in known:
            raise SystemExit(f"Unknown activation '{a}'")
        if key not in out:
            out.append(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional HH Poincaré section at x=0")
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["GaborActivation", "mySin"],
        help="Activations to plot (default: Gabor + mySin contrast)",
    )
    parser.add_argument(
        "--tier",
        default="canonical",
        choices=list(POINCARE_TIERS),
        help="canonical = main pipeline IC + ../models/; else tier IC + studies/models/",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override single checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--n-dense", type=int, default=8000, help="Time samples for section")
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    activations = resolve_activations(args.activations)

    if args.tier == "canonical":
        out_root = EXP_STUDIES / "poincare" / "canonical"
    else:
        out_root = EXP_STUDIES / "poincare" / args.tier
        X0_tier, E_found = tier_initial_state(args.tier)
        print(
            f"[poincare] tier={args.tier} target E={ENERGY_TIERS[args.tier]:.5f} "
            f"found E={E_found:.5f} X0={X0_tier[1:]}",
            flush=True,
        )
    out_root.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        plot_poincare_for_activation(
            "custom",
            args.checkpoint,
            out_root / "custom",
            args.n_dense,
            device,
            tier=args.tier,
        )
        return

    X0 = resolve_X0(args.tier)
    for act in activations:
        ck = default_checkpoint(act, tier=args.tier)
        plot_poincare_for_activation(
            act,
            ck,
            out_root / activation_slug(act),
            args.n_dense,
            device,
            tier=args.tier,
            X0=X0,
        )


if __name__ == "__main__":
    main()
