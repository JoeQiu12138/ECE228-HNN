#!/usr/bin/env python3
"""
Phase 3 — coupled nonlinear oscillators: train, evaluate, compare with Phase 2 HH.

Usage (from src/Phase3/coupled_oscillator/):
  python3 p3_train_eval.py --eval-only --device auto
  python3 p3_train_eval.py --device cuda --train all
  python3 p3_train_eval.py --device cpu --quick --train mysin-long gabor-long
  python3 p3_train_eval.py --smoke
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

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
EXP_DATA = REPO / "exp_data" / "coupled_oscillator"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PHASE2_RESULTS = REPO / "src" / "Phase2" / "HHsystem" / "results" / "hh_activation_comparison.json"

from p3_model import (  # noqa: E402
    evaluate_coupled_model,
    load_model_from_checkpoint,
    parametric_solutions,
    resolve_device,
    train_coupled_model,
)
from physics_utils import coupled_exact, coupled_solution, energy

X0 = [0.0, 0.5, -0.3, 0.25, 0.15, 0.5]
NEURONS = 80
N_TRAIN = 500
SHORT_T = 6 * np.pi
LONG_T = 12 * np.pi
SHORT_EPOCHS = 20_000
LONG_EPOCHS = 50_000
SHORT_EPOCHS_QUICK = 2_000
LONG_EPOCHS_QUICK = 5_000
SHORT_LR = 8e-3
LONG_LR = 5e-3
LOG_EVERY = 500

ACTIVATIONS = ("mySin", "AdaptiveSin", "GaborActivation")


def default_runs() -> list[dict]:
    return [
        {
            "activation": "mySin",
            "horizon": "long",
            "checkpoint": MODELS_DIR / "model_coupled_mySin_longtime.zip",
            "short_checkpoint": MODELS_DIR / "model_coupled_mySin_short.zip",
            "experiment_dir": EXP_DATA / "baseline",
        },
        {
            "activation": "AdaptiveSin",
            "horizon": "long",
            "checkpoint": MODELS_DIR / "model_coupled_adaptiveSin_longtime.zip",
            "short_checkpoint": MODELS_DIR / "model_coupled_adaptiveSin_short.zip",
            "experiment_dir": EXP_DATA / "adaptiveSin",
        },
        {
            "activation": "GaborActivation",
            "horizon": "long",
            "checkpoint": MODELS_DIR / "model_coupled_Gabor_longtime.zip",
            "short_checkpoint": MODELS_DIR / "model_coupled_Gabor_short.zip",
            "experiment_dir": EXP_DATA / "Gabor",
        },
    ]


def run_train_slug(run: dict) -> str:
    slug = {"mySin": "mysin", "AdaptiveSin": "adaptivesin", "GaborActivation": "gabor"}[
        run["activation"]
    ]
    return f"{slug}-long"


def resolve_train_targets(requested: list[str], runs: list[dict]) -> list[dict]:
    by_slug = {run_train_slug(r): r for r in runs}
    aliases = {"mysin": "mysin-long", "adaptivesin": "adaptivesin-long", "gabor": "gabor-long"}
    if any(t.lower() == "all" for t in requested):
        return list(runs)
    out = []
    for raw in requested:
        key = aliases.get(raw.lower().replace("_", "-"), raw.lower().replace("_", "-"))
        if key not in by_slug:
            raise SystemExit(f"Unknown --train target '{raw}'. Choose: {', '.join(by_slug)}, all")
        if by_slug[key] not in out:
            out.append(by_slug[key])
    return out


def _warm_start_long(
    activation: str,
    ck_short: Path,
    ck_long: Path,
    short_epochs: int,
    long_epochs: int,
    log_every: int,
    device: str | None,
):
    if not ck_short.exists():
        train_coupled_model(
            activation,
            X0,
            SHORT_T,
            N_TRAIN,
            NEURONS,
            short_epochs,
            SHORT_LR,
            ck_short,
            load_weights=False,
            perturb_sigma=0.03,
            log_every=log_every,
            stage_label=f"{activation} | short {short_epochs} ep",
            device=device,
        )
    shutil.copy2(ck_short, ck_long)
    return train_coupled_model(
        activation,
        X0,
        LONG_T,
        N_TRAIN,
        NEURONS,
        long_epochs,
        LONG_LR,
        ck_long,
        load_weights=True,
        perturb_sigma=0.3,
        log_every=log_every,
        stage_label=f"{activation} | long {long_epochs} ep",
        device=device,
    )


def save_plots(model, t_max: float, n_test: int, out_dir: Path) -> None:
    dev = next(model.parameters()).device
    t0, x10, x20, p10, p20, beta = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    t_test = torch.linspace(t0, t_max, n_test, device=dev).reshape(-1, 1)
    t_test.requires_grad = True
    x1, x2, p1, p2 = parametric_solutions(t_test, model, X0)
    x1_np = x1.detach().cpu().numpy()[:, 0]
    x2_np = x2.detach().cpu().numpy()[:, 0]
    p1_np = p1.detach().cpu().numpy()[:, 0]
    p2_np = p2.detach().cpu().numpy()[:, 0]
    E_net = energy(x1_np, x2_np, p1_np, p2_np, beta)
    t_net = t_test.detach().cpu().numpy()[:, 0]

    t_num = np.linspace(t0, t_max, n_test)
    E0, E_ex = coupled_exact(n_test, x10, x20, p10, p20, beta)
    x1n, x2n, p1n, p2n = coupled_solution(n_test, t_num, x10, x20, p10, p20, beta)

    prefix = "CoupledOscillator"
    lw = 2
    plt.figure(figsize=(10, 8))
    for i, (name, gt, nn) in enumerate(
        [("x1", x1n, x1_np), ("x2", x2n, x2_np), ("p1", p1n, p1_np), ("p2", p2n, p2_np)]
    ):
        plt.subplot(2, 2, i + 1)
        plt.plot(t_num, gt, "-g", linewidth=lw, label="truth")
        plt.plot(t_net, nn, "--b", label="NN")
        plt.ylabel(name)
        plt.xlabel("t")
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_trajectories.png")
    plt.close()

    d1, d2, dp1, dp2 = x1n - x1_np, x2n - x2_np, p1n - p1_np, p2n - p2_np
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t_net, d1, "b")
    plt.ylabel(r"$\delta_{x_1}$")
    plt.xlabel("t")
    plt.subplot(2, 2, 2)
    plt.plot(t_net, E_ex, "-g")
    plt.plot(t_net, E_net, "--b")
    plt.ylabel("E")
    plt.xlabel("t")
    plt.subplot(2, 2, 3)
    plt.plot(t_net, d2, "b")
    plt.ylabel(r"$\delta_{x_2}$")
    plt.xlabel("t")
    plt.subplot(2, 2, 4)
    plt.plot(d1, dp1, "b")
    plt.ylabel(r"$\delta_{p_1}$")
    plt.xlabel(r"$\delta_{x_1}$")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_error.png")
    plt.close()

    np.savetxt(out_dir / "dx1.txt", d1)
    np.savetxt(out_dir / "dx2.txt", d2)
    np.savetxt(out_dir / "dp1.txt", dp1)
    np.savetxt(out_dir / "dp2.txt", dp2)
    np.savetxt(out_dir / "E.txt", E_net)
    np.savetxt(out_dir / "t_net.txt", t_net)


def export_run(run: dict, t_max: float, device: str | None, loss_hist: list | None = None) -> None:
    ck = run["checkpoint"]
    out_dir = Path(run["experiment_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_model_from_checkpoint(ck, NEURONS, device=device)
    save_plots(model, t_max, N_TRAIN, out_dir)
    if loss_hist is not None and len(loss_hist):
        np.savetxt(out_dir / "loss.txt", np.asarray(loss_hist))


def _qualitative(metrics: dict) -> str:
    err, drift = metrics["max_trajectory_error"], metrics["max_energy_drift"]
    if err < 5e-3 and drift < 1e-3:
        return "strong match to ground truth"
    if err < 2e-2 and drift < 5e-3:
        return "acceptable long-time rollout"
    return "visible drift or phase error"


def evaluate_run(run: dict, device: str | None) -> dict:
    model, ck_loss = load_model_from_checkpoint(run["checkpoint"], NEURONS, device=device)
    metrics = evaluate_coupled_model(model, X0, LONG_T, N_TRAIN, device=device)
    return {
        "system": "Coupled nonlinear oscillators",
        "activation": run["activation"],
        "horizon": run["horizon"],
        "final_train_loss": ck_loss if ck_loss is not None else float("nan"),
        **metrics,
        "qualitative_outcome": _qualitative(metrics),
        "checkpoint": str(run["checkpoint"]),
        "experiment_dir": str(run["experiment_dir"]),
    }


def write_p3_tables(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = RESULTS_DIR / "p3_activation_comparison.md"
    lines = [
        "# Phase 3 — coupled oscillator activation comparison",
        "",
        "System: two coupled nonlinear oscillators, "
        f"$H = (p_1^2+p_2^2+x_1^2+x_2^2)/2 + \\beta x_1^2 x_2^2$, $\\beta={X0[5]}$.",
        f"Long-time: $t \\in [0, {LONG_T/np.pi:.0f}\\pi]$, {N_TRAIN} points, warm-start from $6\\pi$.",
        "",
        "| Activation | Final loss | Max traj. err | Mean traj. err | Max E drift | Mean E drift | Outcome |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['activation']} | {r['final_train_loss']:.3e} | "
            f"{r['max_trajectory_error']:.3e} | {r['mean_trajectory_error']:.3e} | "
            f"{r['max_energy_drift']:.3e} | {r['mean_energy_drift']:.3e} | {r['qualitative_outcome']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "p3_activation_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (RESULTS_DIR / "p3_activation_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {md}", flush=True)


def write_phase2_vs_phase3(p3_rows: list[dict]) -> None:
    """Merge Phase 2 HH long-time rows with Phase 3 rows."""
    hh_rows = []
    if PHASE2_RESULTS.exists():
        raw = json.loads(PHASE2_RESULTS.read_text(encoding="utf-8"))
        for r in raw:
            if r.get("horizon") != "long" or r.get("activation") not in ACTIVATIONS:
                continue
            hh_rows.append(
                {
                    "system": "Hénon-Heiles",
                    "activation": r["activation"],
                    "horizon": r["horizon"],
                    "final_train_loss": r.get("final_train_loss"),
                    "max_trajectory_error": r.get("max_trajectory_error"),
                    "mean_trajectory_error": r.get("mean_trajectory_error"),
                    "max_energy_drift": r.get("max_energy_drift"),
                    "mean_energy_drift": r.get("mean_energy_drift"),
                    "qualitative_outcome": r.get("qualitative_outcome"),
                }
            )

    combined = hh_rows + [
        {
            "system": r["system"],
            "activation": r["activation"],
            "horizon": r["horizon"],
            "final_train_loss": r["final_train_loss"],
            "max_trajectory_error": r["max_trajectory_error"],
            "mean_trajectory_error": r["mean_trajectory_error"],
            "max_energy_drift": r["max_energy_drift"],
            "mean_energy_drift": r["mean_energy_drift"],
            "qualitative_outcome": r["qualitative_outcome"],
        }
        for r in p3_rows
    ]

    out_md = RESULTS_DIR / "phase2_vs_phase3_comparison.md"
    lines = [
        "# Phase 2 (HH) vs Phase 3 (coupled oscillators)",
        "",
        "Long-time rows only; activations: mySin, AdaptiveSin, GaborActivation.",
        "",
        "| System | Activation | Final loss | Max traj. err | Mean traj. err | Max E drift | Outcome |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in combined:
        fl = r["final_train_loss"]
        fl_s = f"{fl:.3e}" if fl is not None and np.isfinite(fl) else "N/A"
        lines.append(
            f"| {r['system']} | {r['activation']} | {fl_s} | "
            f"{r['max_trajectory_error']:.3e} | {r['mean_trajectory_error']:.3e} | "
            f"{r['max_energy_drift']:.3e} | {r['qualitative_outcome']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "phase2_vs_phase3_comparison.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_md}", flush=True)


def smoke_test(device: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ck = MODELS_DIR / "model_coupled_smoke_mySin.zip"
    if ck.exists():
        ck.unlink()
    print(f"[smoke] train+eval on {device} ...", flush=True)
    model, loss_hist, _ = train_coupled_model(
        "mySin",
        X0,
        SHORT_T,
        N_TRAIN,
        NEURONS,
        50,
        SHORT_LR,
        ck,
        load_weights=False,
        log_every=25,
        stage_label=f"smoke mySin ({device})",
        device=device,
    )
    evaluate_coupled_model(model, X0, SHORT_T, N_TRAIN, device=device)
    run = {"checkpoint": ck, "experiment_dir": EXP_DATA / "_smoke"}
    export_run(run, SHORT_T, device, loss_hist)
    if (EXP_DATA / "_smoke").exists():
        shutil.rmtree(EXP_DATA / "_smoke")
    ck.unlink(missing_ok=True)
    print("[smoke] OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 coupled oscillator pipeline")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train", nargs="+", metavar="TARGET")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--export-all", action="store_true", help="Export plots/txt for existing checkpoints")
    args = parser.parse_args()

    if args.smoke:
        for dev in ("cpu", "cuda"):
            if dev == "cuda" and not torch.cuda.is_available():
                print("[smoke] skip CUDA", flush=True)
                continue
            smoke_test(dev)
        return

    device_arg = None if args.device == "auto" else args.device
    short_ep = SHORT_EPOCHS_QUICK if args.quick else SHORT_EPOCHS
    long_ep = LONG_EPOCHS_QUICK if args.quick else LONG_EPOCHS
    log_every = args.log_every

    torch.manual_seed(228)
    np.random.seed(228)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DATA.mkdir(parents=True, exist_ok=True)

    runs = default_runs()

    if args.export_all:
        for run in runs:
            if run["checkpoint"].exists():
                print(f">>> Export {run['activation']} ...", flush=True)
                export_run(run, LONG_T, device_arg)

    if args.train:
        for run in resolve_train_targets(args.train, runs):
            act = run["activation"]
            model, loss_hist, _ = _warm_start_long(
                act,
                Path(run["short_checkpoint"]),
                Path(run["checkpoint"]),
                short_ep,
                long_ep,
                log_every,
                device_arg,
            )
            export_run(run, LONG_T, device_arg, loss_hist)

    rows = []
    for run in runs:
        ck = run["checkpoint"]
        if not ck.exists():
            print(f"SKIP (no ckpt): {ck.name}", flush=True)
            continue
        print(f"[eval] {run['activation']} <- {ck.name}", flush=True)
        row = evaluate_run(run, device_arg)
        rows.append(row)
        print(
            f"       loss={row['final_train_loss']:.3e} max_err={row['max_trajectory_error']:.3e}",
            flush=True,
        )

    if rows:
        write_p3_tables(rows)
        write_phase2_vs_phase3(rows)
    print(">>> Phase 3 done.", flush=True)


if __name__ == "__main__":
    main()
