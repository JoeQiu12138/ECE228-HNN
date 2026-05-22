#!/usr/bin/env python3
"""
Optional HH study: energy-tier comparison (low / medium / high chaotic).

NOT part of the main proposal pipeline (hh_train_eval.py).
Checkpoints and figures go to studies/models/ and exp_data/HH_studies/.

Usage (from src/Phase2/HHsystem/studies/):
  python3 energy_tier_study.py --smoke
  python3 energy_tier_study.py --device cuda --quick --train mysin --tier low_regular
  python3 energy_tier_study.py --device cuda --train all --tier all
  python3 energy_tier_study.py --eval-only --device auto
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HH_ROOT = Path(__file__).resolve().parent.parent
if str(HH_ROOT) not in sys.path:
    sys.path.insert(0, str(HH_ROOT))

from hh_model import (  # noqa: E402
    evaluate_hh_model,
    load_model_from_checkpoint,
    parametricSolutions,
    resolve_device,
    train_hh_model,
)
from hh_physics_utils import HHsolution, energy

from hh_study_common import (  # noqa: E402
    ACTIVATIONS_ALL,
    ENERGY_TIERS,
    ESCAPE_E,
    EXP_STUDIES,
    LONG_T,
    MODELS_STUDIES,
    N_TRAIN,
    RESULTS_STUDIES,
    SHORT_T,
    activation_slug,
    find_initial_state,
    hh_energy_scalar,
    study_checkpoint,
    study_experiment_dir,
    tier_X0_table,
)

SHORT_EPOCHS = 20_000
LONG_EPOCHS = 50_000
SHORT_EPOCHS_QUICK = 2_000
LONG_EPOCHS_QUICK = 5_000
SHORT_LR = 8e-3
LONG_LR = 5e-3
LOG_EVERY = 500
NEURONS = 80


def _warm_start(
    activation: str,
    X0: list[float],
    ck_short: Path,
    ck_long: Path,
    short_ep: int,
    long_ep: int,
    device: str | None,
):
    if not ck_short.exists():
        train_hh_model(
            activation,
            X0,
            SHORT_T,
            N_TRAIN,
            NEURONS,
            short_ep,
            SHORT_LR,
            ck_short,
            load_weights=False,
            perturb_sigma=0.03,
            log_every=LOG_EVERY,
            stage_label=f"{activation} tier-short",
            device=device,
        )
    shutil.copy2(ck_short, ck_long)
    return train_hh_model(
        activation,
        X0,
        LONG_T,
        N_TRAIN,
        NEURONS,
        long_ep,
        LONG_LR,
        ck_long,
        load_weights=True,
        perturb_sigma=0.3,
        log_every=LOG_EVERY,
        stage_label=f"{activation} tier-long",
        device=device,
    )


def export_plots(model, X0: list[float], out_dir: Path, device: str | None) -> None:
    dev = resolve_device(device)
    model = model.to(dev)
    t0, x0, y0, px0, py0, lam = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    t_num = np.linspace(t0, LONG_T, N_TRAIN)
    xn, yn, pxn, pyn = HHsolution(N_TRAIN, t_num, x0, y0, px0, py0, lam)
    t_t = torch.linspace(t0, LONG_T, N_TRAIN, device=dev).reshape(-1, 1)
    t_t.requires_grad = True
    x, y, px, py = parametricSolutions(t_t, model, X0)
    x_np = x.detach().cpu().numpy()[:, 0]
    y_np = y.detach().cpu().numpy()[:, 0]

    plt.figure(figsize=(8, 4))
    plt.plot(t_num, xn, "-g", label="truth")
    plt.plot(t_num, x_np, "--b", label="NN")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tier_trajectory_x.png")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(xn, yn, "-g", alpha=0.7, label="truth")
    plt.plot(x_np, y_np, "--b", alpha=0.7, label="NN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tier_phase_xy.png")
    plt.close()


def _qualitative(metrics: dict) -> str:
    err, drift = metrics["max_trajectory_error"], metrics["max_energy_drift"]
    if err < 5e-3 and drift < 1e-3:
        return "strong"
    if err < 0.15 and drift < 0.05:
        return "acceptable"
    return "poor / divergent"


def resolve_tiers(raw: list[str]) -> list[str]:
    if any(t.lower() == "all" for t in raw):
        return list(ENERGY_TIERS.keys())
    out = []
    for t in raw:
        if t not in ENERGY_TIERS:
            raise SystemExit(f"Unknown tier '{t}'. Choose: {', '.join(ENERGY_TIERS)}, all")
        out.append(t)
    return out


def resolve_activations(raw: list[str]) -> list[str]:
    aliases = {
        "mysin": "mySin",
        "adaptivesin": "AdaptiveSin",
        "dualsin": "DualAdaptiveSin",
        "gabor": "GaborActivation",
        "polynomial": "LearnablePolynomial",
    }
    if any(a.lower() == "all" for a in raw):
        return list(ACTIVATIONS_ALL)
    out = []
    for a in raw:
        key = aliases.get(a.lower(), a)
        if key not in ACTIVATIONS_ALL:
            raise SystemExit(f"Unknown activation '{a}'")
        if key not in out:
            out.append(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional HH energy-tier study")
    parser.add_argument("--train", nargs="+", metavar="TARGET")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--tier", nargs="+", default=["all"], help="low_regular, medium_baseline, high_chaotic, all")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    short_ep = SHORT_EPOCHS_QUICK if args.quick else SHORT_EPOCHS
    long_ep = LONG_EPOCHS_QUICK if args.quick else LONG_EPOCHS

    MODELS_STUDIES.mkdir(parents=True, exist_ok=True)
    EXP_STUDIES.mkdir(parents=True, exist_ok=True)
    RESULTS_STUDIES.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        X0, _ = find_initial_state(0.08, seed=99)
        ck = MODELS_STUDIES / "smoke_tier.zip"
        if ck.exists():
            ck.unlink()
        train_hh_model(
            "mySin",
            X0,
            SHORT_T,
            N_TRAIN,
            NEURONS,
            30,
            SHORT_LR,
            ck,
            log_every=10,
            stage_label="smoke tier",
            device=device,
        )
        ck.unlink(missing_ok=True)
        print("[smoke] energy tier OK", flush=True)
        return

    tier_x0 = tier_X0_table()
    tiers = resolve_tiers(args.tier)
    activations = resolve_activations(args.train) if args.train else list(ACTIVATIONS_ALL)

    if args.train and not args.eval_only:
        for tier in tiers:
            X0 = tier_x0[tier]
            for act in activations:
                if act == "LearnablePolynomial":
                    print(f"SKIP train polynomial on tier {tier}", flush=True)
                    continue
                ck_long = study_checkpoint(tier, act)
                ck_short = ck_long.with_name(ck_long.name.replace("longtime", "short"))
                print(f"\n>>> train {act} @ {tier}", flush=True)
                model, _, _ = _warm_start(act, X0, ck_short, ck_long, short_ep, long_ep, device)
                out = study_experiment_dir(tier, act)
                out.mkdir(parents=True, exist_ok=True)
                export_plots(model, X0, out, device)

    rows = []
    for tier in tiers:
        X0 = tier_x0[tier]
        Et = ENERGY_TIERS[tier]
        E0 = hh_energy_scalar(X0[1], X0[2], X0[3], X0[4], X0[5])
        for act in ACTIVATIONS_ALL:
            ck = study_checkpoint(tier, act)
            if not ck.exists():
                if act == "LearnablePolynomial":
                    continue
                print(f"SKIP eval (no ckpt): {tier}/{act}", flush=True)
                continue
            print(f"[eval] {tier} / {act}", flush=True)
            model, ck_loss = load_model_from_checkpoint(ck, NEURONS, device=device)
            m = evaluate_hh_model(model, X0, LONG_T, N_TRAIN, device=device)
            rows.append(
                {
                    "tier": tier,
                    "E_target": Et,
                    "E0": E0,
                    "escape_threshold": ESCAPE_E,
                    "activation": act,
                    "final_train_loss": ck_loss,
                    **m,
                    "qualitative": _qualitative(m),
                    "checkpoint": str(ck),
                }
            )

    if not rows and not args.train:
        print("No checkpoints found. Run with --train first.", flush=True)
        return
    if not rows:
        return

    md = RESULTS_STUDIES / "energy_tier_comparison.md"
    lines = [
        "# Optional study: HH energy tiers",
        "",
        f"Escape energy $E_{{esc}}=1/6\\approx{ESCAPE_E:.4f}$. Not part of main `hh_train_eval.py` table.",
        "",
        "`medium_baseline` matches the canonical HH energy scale, but uses a tier-generated initial condition and study checkpoints. Interpret this table as sensitivity analysis, not as a reproduction of the canonical HH row.",
        "",
        "| Tier | E_target | E0 | Activation | Final loss | Max traj err | Max E drift | Qualitative |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['tier']} | {r['E_target']:.4f} | {r['E0']:.4f} | {r['activation']} | "
            f"{r['final_train_loss']:.3e} | {r['max_trajectory_error']:.3e} | "
            f"{r['max_energy_drift']:.3e} | {r['qualitative']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_STUDIES / "energy_tier_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (RESULTS_STUDIES / "energy_tier_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {md}", flush=True)


if __name__ == "__main__":
    main()
