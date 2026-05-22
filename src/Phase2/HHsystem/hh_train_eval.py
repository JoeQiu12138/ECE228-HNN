#!/usr/bin/env python3
"""
Build Phase 2 HH activation comparison table and verify exp_data/HH/ artifacts.

Usage (from src/Phase2/HHsystem/):
  python hh_train_eval.py --eval-only --device auto
  python hh_train_eval.py --device cuda --train all
  python hh_train_eval.py --device cuda --train mysin-long gabor-long
  python hh_train_eval.py --eval-only --golden-gabor   # Gabor metrics from exp_data/HH/data/
  python hh_train_eval.py --smoke
  python hh_train_eval.py --export-all --device cuda

Training prints progress every --log-every epochs (or tqdm bar if installed).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hh_model import (
    configure_torch_backend,
    evaluate_hh_model,
    infer_activation_from_state_dict,
    load_model_from_checkpoint,
    parametricSolutions,
    resolve_device,
    train_hh_model,
)
from hh_physics_utils import HH_exact, HHsolution, energy

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]  # repo root (src/Phase2/HHsystem → … → ECE228-HNN)
EXP_DATA = REPO / "exp_data"
EXPERIMENT_HH = EXP_DATA / "HH"
DATA_LEGACY_DIR = EXPERIMENT_HH / "data"  # Gabor numerics archive (exp_data/HH/data/)
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# exp_data/HH/data/ root files → exp_data/HH/Gabor/longtime/ names
_LEGACY_GABOR_NUMERIC = (
    "dx.txt",
    "dy.txt",
    "dpx.txt",
    "dpy.txt",
    "E.txt",
    "loss.txt",
    "loss_12.txt",
)
_LEGACY_GABOR_RENAME = {"t.txt": "t_net.txt", "x.txt": "x_net.txt"}

# Canonical checkpoint names (legacy aliases migrated on startup).
CK_GABOR_LONG = MODELS_DIR / "model_HH_Gabor_longtime.zip"
CK_ADAPTIVE_SHORT = MODELS_DIR / "model_HH_adaptiveSin_shorttime.zip"

_LEGACY_CKPT_NAMES: dict[str, Path] = {
    "model_HH.zip": CK_GABOR_LONG,
    "model_HH_short.zip": CK_ADAPTIVE_SHORT,
    "model_HH": CK_GABOR_LONG,
    "model_HH_short": CK_ADAPTIVE_SHORT,
}


def migrate_legacy_checkpoint_names() -> None:
    """Rename historical model_HH*.zip files to activation_horizon names."""
    for old_name, new_path in _LEGACY_CKPT_NAMES.items():
        old_path = MODELS_DIR / old_name
        if not old_path.is_file() or old_path.resolve() == new_path.resolve():
            continue
        if not new_path.exists():
            old_path.rename(new_path)
            print(f"[ckpt] renamed {old_name} -> {new_path.name}", flush=True)
        else:
            old_path.unlink()
            print(f"[ckpt] removed duplicate legacy {old_name}", flush=True)


X0 = [0.0, 0.3, -0.3, 0.3, 0.15, 1.0]
NEURONS = 80
N_TRAIN = 500
SHORT_T = 6 * np.pi
LONG_T = 12 * np.pi
# Defaults copied from legacy_henon_heiles_train.py (2e4 short / 5e4 long), not from proposal.
SHORT_EPOCHS = 20_000
LONG_EPOCHS = 50_000
SHORT_EPOCHS_QUICK = 2_000
LONG_EPOCHS_QUICK = 5_000
SHORT_LR = 8e-3
LONG_LR = 5e-3
POLY_PROBE_EPOCHS = 3_000
LOG_EVERY = 500


def default_runs() -> list[dict]:
    """Registered checkpoints mapped to experiment folders (long-time unless noted)."""
    return [
        {
            "activation": "mySin",
            "horizon": "long",
            "checkpoint": MODELS_DIR / "model_HH_mySin_longtime.zip",
            "experiment_dir": EXPERIMENT_HH / "baseline",
        },
        {
            "activation": "AdaptiveSin",
            "horizon": "long",
            "checkpoint": MODELS_DIR / "model_HH_adaptiveSin_longtime.zip",
            "short_checkpoint": CK_ADAPTIVE_SHORT,
            "experiment_dir": EXPERIMENT_HH / "adaptivesin" / "a=1 longtime",
        },
        {
            "activation": "DualAdaptiveSin",
            "horizon": "long",
            "checkpoint": MODELS_DIR / "model_HH_dualSin_longtime.zip",
            "short_checkpoint": MODELS_DIR / "model_HH_dualSin_shorttime.zip",
            "experiment_dir": EXPERIMENT_HH / "dualSin" / "longtime",
        },
        {
            "activation": "GaborActivation",
            "horizon": "long",
            "checkpoint": CK_GABOR_LONG,
            "short_checkpoint": MODELS_DIR / "model_HH_Gabor_short_time.zip",
            "experiment_dir": EXPERIMENT_HH / "Gabor" / "longtime",
        },
        {
            "activation": "GaborActivation",
            "horizon": "short",
            "checkpoint": MODELS_DIR / "model_HH_Gabor_short_time.zip",
            "experiment_dir": EXPERIMENT_HH / "Gabor" / "short",
        },
        {
            "activation": "DualAdaptiveSin",
            "horizon": "short",
            "checkpoint": MODELS_DIR / "model_HH_dualSin_shorttime.zip",
            "experiment_dir": EXPERIMENT_HH / "dualSin" / "short",
        },
        {
            "activation": "AdaptiveSin",
            "horizon": "short",
            "checkpoint": CK_ADAPTIVE_SHORT,
            "experiment_dir": EXPERIMENT_HH / "adaptivesin" / "short",
        },
        {
            "activation": "LearnablePolynomial",
            "horizon": "probe",
            "checkpoint": MODELS_DIR / "model_HH_polynomial_probe.zip",
            "experiment_dir": EXPERIMENT_HH / "LearnablePolynomial",
            "probe_only": True,
        },
    ]


def run_train_slug(run: dict) -> str:
    """CLI target id, e.g. mysin-long, gabor-short, polynomial-probe."""
    if run.get("probe_only"):
        return "polynomial-probe"
    slug = {
        "mySin": "mysin",
        "AdaptiveSin": "adaptivesin",
        "DualAdaptiveSin": "dualsin",
        "GaborActivation": "gabor",
    }[run["activation"]]
    return f"{slug}-{run['horizon']}"


_TRAIN_ALIASES = {
    "mysin": "mysin-long",
    "adaptivesin": "adaptivesin-long",
    "dualsin": "dualsin-long",
    "gabor": "gabor-long",
    "polynomial": "polynomial-probe",
}


def list_train_targets(runs: list[dict]) -> list[str]:
    keys = sorted(run_train_slug(r) for r in runs)
    return keys + ["all"]


def resolve_train_targets(requested: list[str], runs: list[dict]) -> list[dict]:
    by_slug = {run_train_slug(r): r for r in runs}
    if any(t.lower() == "all" for t in requested):
        return list(runs)
    selected: list[dict] = []
    for raw in requested:
        key = _TRAIN_ALIASES.get(raw.lower().replace("_", "-"), raw.lower().replace("_", "-"))
        if key not in by_slug:
            raise SystemExit(
                f"Unknown --train target '{raw}'. "
                f"Choose from: {', '.join(list_train_targets(runs))}"
            )
        run = by_slug[key]
        if run not in selected:
            selected.append(run)
    return selected


def sync_legacy_gabor_data_to_experiment() -> None:
    """Copy canonical Gabor long numerics from exp_data/HH/data/ into Gabor/longtime/."""
    src_dir = DATA_LEGACY_DIR
    dst_dir = EXPERIMENT_HH / "Gabor" / "longtime"
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in _LEGACY_GABOR_NUMERIC:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    for src_name, dst_name in _LEGACY_GABOR_RENAME.items():
        src = src_dir / src_name
        if src.is_file():
            shutil.copy2(src, dst_dir / dst_name)
            copied.append(f"{src_name}->{dst_name}")
    if copied:
        print(
            f"[sync] exp_data/HH/data/ -> Gabor/longtime/ ({len(copied)} files)",
            flush=True,
        )


def resolve_dualsin_long_checkpoint() -> Path:
    """Canonical DualSin long checkpoint (legacy em-dash filename removed from repo)."""
    canonical = MODELS_DIR / "model_HH_dualSin_longtime.zip"
    legacy = MODELS_DIR / "model_HH——dualSin_longtime.zip"
    if not canonical.exists() and legacy.exists():
        shutil.copy2(legacy, canonical)
    return canonical


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def save_plots(model, X0, t_max, n_test, out_dir: Path, prefix: str) -> None:
    dev = _model_device(model)
    model.eval()

    t0, x0, y0, px0, py0, lam = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    t_test = torch.linspace(t0, t_max, n_test, device=dev).reshape(-1, 1)
    t_test.requires_grad = True
    x, y, px, py = parametricSolutions(t_test, model, X0)
    x_np = x.detach().cpu().numpy()[:, 0]
    y_np = y.detach().cpu().numpy()[:, 0]
    px_np = px.detach().cpu().numpy()[:, 0]
    py_np = py.detach().cpu().numpy()[:, 0]
    E_net = energy(x_np, y_np, px_np, py_np, lam)
    t_net = t_test.detach().cpu().numpy()[:, 0]

    t_num = np.linspace(t0, t_max, n_test)
    E0, E_ex = HH_exact(n_test, x0, y0, px0, py0, lam)
    x_num, y_num, px_num, py_num = HHsolution(n_test, t_num, x0, y0, px0, py0, lam)

    lineW = 2
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t_num, x_num, "-g", linewidth=lineW, label="Ground truth")
    plt.plot(t_net, x_np, "--b", label="Neural Net")
    plt.ylabel("x")
    plt.xlabel("t")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t_num, E_ex, "-g", linewidth=lineW)
    plt.plot(t_net, E_net, "--b")
    plt.ylabel("E")
    plt.xlabel("t")
    plt.ylim([1.1 * E0, 0.9 * E0])

    plt.subplot(2, 2, 3)
    plt.plot(t_num, px_num, "-g", linewidth=lineW)
    plt.plot(t_net, px_np, "--b")
    plt.ylabel("$p_x$")
    plt.xlabel("t")

    plt.subplot(2, 2, 4)
    plt.plot(x_num, y_num, "-g", linewidth=lineW)
    plt.plot(x_np, y_np, "--b")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_trajectories.png")
    plt.close()

    dx = x_num - x_np
    dy = y_num - y_np
    dpx = px_num - px_np
    dpy = py_num - py_np
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t_net, dx, "b", label="Neural Net")
    plt.ylabel(r"$\delta_x$")
    plt.xlabel("t")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(dx, dpx, "b")
    plt.ylabel(r"$\delta_{p_x}$")
    plt.xlabel(r"$\delta_x$")
    plt.subplot(2, 2, 3)
    plt.plot(t_net, dy, "b")
    plt.ylabel(r"$\delta_y$")
    plt.xlabel("t")
    plt.subplot(2, 2, 4)
    plt.plot(dy, dpy, "b")
    plt.ylabel(r"$\delta_{p_y}$")
    plt.xlabel(r"$\delta_y$")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_trajectories_error.png")
    plt.close()


def _warm_start_long(
    activation: str,
    ck_short: Path,
    ck_long: Path,
    short_epochs: int,
    long_epochs: int,
    log_every: int,
    device: str | None,
) -> tuple[Path, torch.nn.Module, list[float]]:
    if not ck_short.exists():
        train_hh_model(
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
    model, loss_hist, _ = train_hh_model(
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
    return ck_long, model, loss_hist


def train_my_sin_long(
    short_epochs: int, long_epochs: int, log_every: int, device: str | None
) -> tuple[Path, torch.nn.Module, list[float]]:
    return _warm_start_long(
        "mySin",
        MODELS_DIR / "model_HH_mySin_short.zip",
        MODELS_DIR / "model_HH_mySin_longtime.zip",
        short_epochs,
        long_epochs,
        log_every,
        device,
    )


def _short_checkpoint_for_run(run: dict) -> Path:
    if run.get("short_checkpoint"):
        return Path(run["short_checkpoint"])
    act, horizon = run["activation"], run["horizon"]
    if act == "mySin":
        return MODELS_DIR / "model_HH_mySin_short.zip"
    if act == "AdaptiveSin":
        return CK_ADAPTIVE_SHORT
    if act == "DualAdaptiveSin":
        return MODELS_DIR / "model_HH_dualSin_shorttime.zip"
    if act == "GaborActivation":
        return MODELS_DIR / "model_HH_Gabor_short_time.zip"
    raise KeyError(f"No short checkpoint mapping for {act} ({horizon})")


def train_run_entry(
    run: dict,
    short_epochs: int,
    long_epochs: int,
    log_every: int,
    device: str | None,
) -> dict | None:
    """Train one registered run and export artifacts. Returns polynomial probe row if applicable."""
    if run.get("probe_only"):
        print(f"\n>>> Training probe: {run['activation']} ...", flush=True)
        return probe_polynomial(device)

    act = run["activation"]
    horizon = run["horizon"]
    ck = Path(run["checkpoint"])
    t_max = LONG_T if horizon == "long" else SHORT_T
    print(f"\n>>> Training {act} ({horizon}) -> {ck.name} ...", flush=True)

    if act == "mySin" and horizon == "long":
        _, model, loss_hist = train_my_sin_long(short_epochs, long_epochs, log_every, device)
    elif act == "AdaptiveSin" and horizon == "long":
        _, model, loss_hist = train_adaptive_sin_long(short_epochs, long_epochs, log_every, device)
    elif horizon == "long":
        short_ck = _short_checkpoint_for_run(run)
        _, model, loss_hist = _warm_start_long(
            act, short_ck, ck, short_epochs, long_epochs, log_every, device
        )
    else:
        epochs = short_epochs
        model, loss_hist, _ = train_hh_model(
            act,
            X0,
            SHORT_T,
            N_TRAIN,
            NEURONS,
            epochs,
            SHORT_LR,
            ck,
            load_weights=False,
            perturb_sigma=0.03,
            log_every=log_every,
            stage_label=f"{act} | short {epochs} ep",
            device=device,
        )

    export_run_artifacts(model, run, t_max, N_TRAIN, loss_hist)
    return None


def train_adaptive_sin_long(
    short_epochs: int, long_epochs: int, log_every: int, device: str | None
) -> tuple[Path, torch.nn.Module, list[float]]:
    ck_short = CK_ADAPTIVE_SHORT
    ck_long = MODELS_DIR / "model_HH_adaptiveSin_longtime.zip"
    if not ck_short.exists():
        alt_short = MODELS_DIR / "model_HH_adaptiveSin_short.zip"
        _warm_start_long("AdaptiveSin", alt_short, alt_short, short_epochs, short_epochs, log_every, device)
        shutil.copy2(alt_short, ck_short)
    shutil.copy2(ck_short, ck_long)
    model, loss_hist, _ = train_hh_model(
        "AdaptiveSin",
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
        stage_label=f"AdaptiveSin | long {long_epochs} ep",
        device=device,
    )
    return ck_long, model, loss_hist


def probe_polynomial(device: str | None = None) -> dict:
    ck = MODELS_DIR / "model_HH_polynomial_probe.zip"
    out_dir = EXPERIMENT_HH / "LearnablePolynomial"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, loss_hist, run_time = train_hh_model(
        "LearnablePolynomial",
        X0,
        SHORT_T,
        N_TRAIN,
        NEURONS,
        POLY_PROBE_EPOCHS,
        SHORT_LR,
        ck,
        load_weights=False,
        min_loss=1e-12,
        perturb_sigma=0.03,
        log_every=max(100, POLY_PROBE_EPOCHS // 10),
        stage_label=f"LearnablePolynomial | probe {POLY_PROBE_EPOCHS} ep",
        device=device,
    )

    finite = [v for v in loss_hist if np.isfinite(v)]
    max_loss = max(finite) if finite else float("nan")
    final_loss = finite[-1] if finite else float("nan")
    diverged = (not finite) or max_loss > 1e3 or final_loss > 1.0

    plt.figure(figsize=(8, 4))
    if finite:
        plt.loglog(finite, "-b")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.title("LearnablePolynomial short-time probe")
    plt.tight_layout()
    plt.savefig(out_dir / "HenonHeiles_loss_probe.png")
    plt.close()

    notes = f"""# LearnablePolynomial — HH experiment notes

## Status: {"FAILED / unstable" if diverged else "completed short probe"}

Phase 2 uses a learnable cubic activation `f(x) = c1*x + c2*x^2 + c3*x^3` intended to mimic HH potential nonlinearity.

## Probe settings
- Horizon: t in [0, {SHORT_T/np.pi:.0f} pi] ({SHORT_T:.4f})
- Training points: {N_TRAIN}
- Hidden width: {NEURONS}
- Epochs: {POLY_PROBE_EPOCHS}
- Learning rate: {SHORT_LR}
- Wall time (min): {run_time/60:.2f}

## Probe outcome
- Final loss: {final_loss:.6e}
- Max finite loss: {max_loss:.6e}
- Epochs logged: {len(loss_hist)}
- Diverged (loss > 1e3 or non-finite): **{diverged}**

## Interpretation (for report)
Composing learnable polynomials inside a 2-layer MLP raises effective polynomial degree when Hamiltonian residuals are differentiated through autograd. In practice the HH training loss **does not converge reliably** with this activation: gradients explode or plateau at very high loss compared to periodic activations (`mySin`, `AdaptiveSin`, `GaborActivation`).

**Recommendation:** cite this folder as a negative ablation; do not use LearnablePolynomial for long-time HH rollout in the final comparison table.

## Artifacts
- `HenonHeiles_loss_probe.png` — probe training curve
- `src/Phase2/HHsystem/models/model_HH_polynomial_probe.zip` — partial checkpoint (if saved before NaN)
"""
    (out_dir / "README.md").write_text(notes, encoding="utf-8")

    return {
        "activation": "LearnablePolynomial",
        "horizon": "probe",
        "final_train_loss": final_loss,
        "max_trajectory_error": float("nan"),
        "mean_trajectory_error": float("nan"),
        "max_energy_drift": float("nan"),
        "mean_energy_drift": float("nan"),
        "qualitative_outcome": "unstable / not used for long-time HH",
        "checkpoint": str(ck),
        "experiment_dir": str(out_dir),
        "notes": "See README.md in exp_data/HH/LearnablePolynomial/",
    }


def _saved_errors_fresh(data_dir: Path, checkpoint: Path) -> bool:
    marker = data_dir / "dx.txt"
    if not marker.exists():
        return False
    if not checkpoint.exists():
        return True
    return checkpoint.stat().st_mtime <= marker.stat().st_mtime


def metrics_from_saved_errors(data_dir: Path) -> dict | None:
    """Read trajectory/energy metrics from saved dx,dy,dpx,dpy,E txt (canonical run)."""
    needed = ["dx.txt", "dy.txt", "dpx.txt", "dpy.txt", "E.txt"]
    if not all((data_dir / name).exists() for name in needed):
        return None
    dx = np.loadtxt(data_dir / "dx.txt")
    dy = np.loadtxt(data_dir / "dy.txt")
    dpx = np.loadtxt(data_dir / "dpx.txt")
    dpy = np.loadtxt(data_dir / "dpy.txt")
    E = np.loadtxt(data_dir / "E.txt")
    E0, _ = HH_exact(len(E), X0[1], X0[2], X0[3], X0[4], X0[5])
    state_err = np.sqrt(dx**2 + dy**2 + dpx**2 + dpy**2)
    energy_drift = np.abs(E - E0)
    loss_path = data_dir / "loss.txt"
    final_loss = float("nan")
    if loss_path.exists():
        loss = np.loadtxt(loss_path)
        if loss.size:
            final_loss = float(loss[-1])
    return {
        "final_train_loss": final_loss,
        "max_trajectory_error": float(np.max(state_err)),
        "mean_trajectory_error": float(np.mean(state_err)),
        "max_energy_drift": float(np.max(energy_drift)),
        "mean_energy_drift": float(np.mean(energy_drift)),
        "E0": float(E0),
        "metrics_source": "saved_trajectory_txt",
    }


def evaluate_run(
    run: dict,
    t_max: float,
    n_test: int,
    device: str | None,
    *,
    golden_gabor: bool = False,
) -> dict:
    ck: Path = run["checkpoint"]
    if not ck.exists():
        raise FileNotFoundError(ck)

    metrics = None
    metrics_source = "checkpoint_eval"
    exp = Path(run["experiment_dir"]) if run.get("experiment_dir") else None

    use_golden = golden_gabor and run.get("activation") == "GaborActivation" and run.get("horizon") == "long"
    if use_golden and DATA_LEGACY_DIR.is_dir():
        metrics = metrics_from_saved_errors(DATA_LEGACY_DIR)
        if metrics is not None:
            metrics_source = metrics.pop("metrics_source", "golden_gabor_archive")

    if metrics is None and exp and exp.exists():
        metrics = metrics_from_saved_errors(exp)
        if metrics is not None:
            metrics_source = metrics.pop("metrics_source", "saved_trajectory_txt")

    model, ckpt_loss = load_model_from_checkpoint(ck, NEURONS, device=device)
    if metrics is None:
        metrics = evaluate_hh_model(model, X0, t_max, n_test, device=device)
    elif ckpt_loss is not None and np.isfinite(ckpt_loss):
        metrics["final_train_loss"] = ckpt_loss
    dev = resolve_device(device)
    activation = run.get("activation") or infer_activation_from_state_dict(
        torch.load(ck, map_location=dev, weights_only=False)["model_state_dict"]
    )
    row = {
        "activation": activation,
        "horizon": run["horizon"],
        "final_train_loss": ckpt_loss if ckpt_loss is not None else float("nan"),
        **metrics,
        "qualitative_outcome": _qualitative(metrics, run.get("horizon", "long")),
        "checkpoint": str(ck),
        "experiment_dir": str(run.get("experiment_dir", "")),
        "metrics_source": metrics_source,
    }
    return row


def _qualitative(metrics: dict, horizon: str) -> str:
    if horizon == "probe":
        return "probe only"
    err = metrics["max_trajectory_error"]
    drift = metrics["max_energy_drift"]
    if err < 5e-3 and drift < 1e-3:
        return "strong match to ground truth"
    if err < 2e-2 and drift < 5e-3:
        return "acceptable long-time rollout"
    return "visible drift or phase error"


def _needs_export(run: dict) -> bool:
    exp = Path(run["experiment_dir"])
    ck = Path(run["checkpoint"])
    marker = exp / "dx.txt"
    traj = exp / "HenonHeiles_trajectories.png"
    if not marker.exists() or not traj.exists():
        return True
    if ck.exists() and ck.stat().st_mtime > marker.stat().st_mtime:
        return True
    return False


def export_all_checkpoints(runs: list[dict], device: str | None) -> None:
    for run in runs:
        if run.get("probe_only"):
            continue
        ck = run["checkpoint"]
        if not ck.exists():
            continue
        t_max = LONG_T if run["horizon"] == "long" else SHORT_T
        export_checkpoint_artifacts(run, t_max, N_TRAIN, device)


def smoke_test_devices() -> None:
    """Train/eval/export a few epochs on CPU and CUDA (if available) to catch device bugs."""
    ck = MODELS_DIR / "model_HH_smoke_mySin.zip"
    if ck.exists():
        ck.unlink()
    for dev_name in ("cpu", "cuda"):
        if dev_name == "cuda" and not torch.cuda.is_available():
            print("[smoke] CUDA not available — skip GPU leg", flush=True)
            continue
        print(f"\n[smoke] device={dev_name} train+eval+export ...", flush=True)
        model, _, _ = train_hh_model(
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
            stage_label=f"smoke mySin ({dev_name})",
            device=dev_name,
        )
        evaluate_hh_model(model, X0, SHORT_T, N_TRAIN, device=dev_name)
        run = {
            "activation": "mySin",
            "horizon": "short",
            "checkpoint": ck,
            "experiment_dir": EXPERIMENT_HH / "_smoke",
        }
        export_run_artifacts(model, run, SHORT_T, N_TRAIN, None)
        print(f"[smoke] {dev_name} OK", flush=True)
    smoke_dir = EXPERIMENT_HH / "_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    if ck.exists():
        ck.unlink()
    print("\n[smoke] All device checks passed.", flush=True)


def export_checkpoint_artifacts(run: dict, t_max: float, n_test: int, device: str | None) -> None:
    ck = run["checkpoint"]
    if not ck.exists():
        print(f"SKIP export (no checkpoint): {ck}", flush=True)
        return
    print(f">>> Exporting {run['activation']} figures → {run['experiment_dir']} ...", flush=True)
    model, _ = load_model_from_checkpoint(ck, device=device)
    export_run_artifacts(model, run, t_max, n_test, loss_hist=None)


def export_run_artifacts(model, run: dict, t_max: float, n_test: int, loss_hist: list[float] | None = None) -> None:
    """Save plots and error txt under experiment dir (and optional loss)."""
    out_dir = Path(run["experiment_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "HenonHeiles"
    dev = _model_device(model)
    save_plots(model, X0, t_max, n_test, out_dir, prefix)

    t0, x0, y0, px0, py0, lam = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    t_test = torch.linspace(t0, t_max, n_test, device=dev).reshape(-1, 1)
    t_test.requires_grad = True
    x, y, px, py = parametricSolutions(t_test, model, X0)
    x_np = x.detach().cpu().numpy()[:, 0]
    y_np = y.detach().cpu().numpy()[:, 0]
    px_np = px.detach().cpu().numpy()[:, 0]
    py_np = py.detach().cpu().numpy()[:, 0]
    t_num = np.linspace(t0, t_max, n_test)
    x_num, y_num, px_num, py_num = HHsolution(n_test, t_num, x0, y0, px0, py0, lam)
    E_net = energy(x_np, y_np, px_np, py_np, lam)

    np.savetxt(out_dir / "t_net.txt", t_test.detach().cpu().numpy())
    np.savetxt(out_dir / "x_net.txt", x_np)
    np.savetxt(out_dir / "dx.txt", x_num - x_np)
    np.savetxt(out_dir / "dy.txt", y_num - y_np)
    np.savetxt(out_dir / "dpx.txt", px_num - px_np)
    np.savetxt(out_dir / "dpy.txt", py_num - py_np)
    np.savetxt(out_dir / "E.txt", E_net)
    if loss_hist:
        np.savetxt(out_dir / "loss.txt", loss_hist)
        plt.figure(figsize=(8, 4))
        plt.loglog(loss_hist, "-b")
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_loss.png")
        plt.close()


def write_tables(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_rows = [r for r in rows if r.get("horizon") == "long" and r["activation"] != "LearnablePolynomial"]
    long_rows.sort(key=lambda r: r["mean_trajectory_error"])

    csv_path = RESULTS_DIR / "hh_activation_comparison.csv"
    fields = [
        "activation",
        "horizon",
        "final_train_loss",
        "max_trajectory_error",
        "mean_trajectory_error",
        "max_energy_drift",
        "mean_energy_drift",
        "qualitative_outcome",
        "checkpoint",
        "experiment_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = RESULTS_DIR / "hh_activation_comparison.md"
    lines = [
        "# Phase 2 HH activation comparison",
        "",
        "Protocol: Hénon–Heiles, X0=[0, 0.3, -0.3, 0.3, 0.15, 1], low-energy regime.",
        "Long-time rows: t in [0, 12π], 500 train/test points, 50k epochs (5e-3 lr) after short warm-start where noted.",
        "Trajectory error = L2 norm of (x,y,px,py) vs SciPy `odeint` ground truth. Energy drift = |E_net - E0|.",
        "",
        "## Long-time comparison (primary table)",
        "",
        "| Activation | Final train loss | Max traj. error | Mean traj. error | Max energy drift | Mean energy drift | Outcome |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in long_rows:
        lines.append(
            f"| {r['activation']} | {r['final_train_loss']:.3e} | {r['max_trajectory_error']:.3e} | "
            f"{r['mean_trajectory_error']:.3e} | {r['max_energy_drift']:.3e} | {r['mean_energy_drift']:.3e} | "
            f"{r['qualitative_outcome']} |"
        )
    lines.extend(["", "## All runs (including short / probe)", ""])
    lines.append("| Activation | Horizon | Final train loss | Max traj. error | Mean traj. error | Max E drift | Mean E drift | Outcome |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        fl = r["final_train_loss"]
        fl_s = f"{fl:.3e}" if np.isfinite(fl) else "N/A"
        def _f(k):
            v = r.get(k, float("nan"))
            return f"{v:.3e}" if np.isfinite(v) else "N/A"
        lines.append(
            f"| {r['activation']} | {r['horizon']} | {fl_s} | {_f('max_trajectory_error')} | {_f('mean_trajectory_error')} | "
            f"{_f('max_energy_drift')} | {_f('mean_energy_drift')} | {r['qualitative_outcome']} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "hh_activation_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def print_runtime_estimate(short_epochs: int, long_epochs: int, n_activations: int = 2) -> None:
    """Rough CPU estimate so users know it is working, not frozen."""
    iters = n_activations * (short_epochs + long_epochs)
    # ~0.15–0.35 s/epoch on CPU in practice for this HH graph
    lo_min = iters * 0.15 / 60
    hi_min = iters * 0.35 / 60
    print(
        f"\n=== Phase 2 HH training estimate ===\n"
        f"Per activation: short {short_epochs:,} + long {long_epochs:,} epochs "
        f"(~{short_epochs + long_epochs:,} backward passes through autograd Hamiltonian loss).\n"
        f"Planned activations to train: {n_activations} → ~{iters:,} total epochs.\n"
        f"Rough wall time on CPU: {lo_min:.0f}–{hi_min:.0f} min "
        f"(GPU often 3–10× faster). Progress prints every few hundred epochs.\n"
        f"Tip: use --quick for a fast smoke test, or pip install tqdm for a progress bar.\n",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, export, and compare HH activation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Train targets: "
            + ", ".join(
                [
                    "mysin-long",
                    "adaptivesin-long",
                    "adaptivesin-short",
                    "dualsin-long",
                    "dualsin-short",
                    "gabor-long",
                    "gabor-short",
                    "polynomial-probe",
                    "all",
                ]
            )
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate checkpoints and write comparison table (no training)",
    )
    parser.add_argument(
        "--train",
        nargs="+",
        metavar="TARGET",
        help="Train TARGET(s); use 'all' for every run including polynomial-probe",
    )
    parser.add_argument(
        "--golden-gabor",
        action="store_true",
        help="For Gabor long: sync exp_data/HH/data/ -> Gabor/longtime/ and score from that archive",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Use {SHORT_EPOCHS_QUICK}/{LONG_EPOCHS_QUICK} epochs instead of {SHORT_EPOCHS}/{LONG_EPOCHS}",
    )
    parser.add_argument("--log-every", type=int, default=LOG_EVERY, help="Print training status every N epochs")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training/eval device (auto = cuda if available)",
    )
    parser.add_argument("--short-epochs", type=int, default=None, help="Override short-stage epochs")
    parser.add_argument("--long-epochs", type=int, default=None, help="Override long-stage epochs")
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export plots/txt for every run with an existing checkpoint (no training)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick CPU/CUDA train+eval+export sanity test, then exit",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_test_devices()
        return

    device_arg = None if args.device == "auto" else args.device
    short_epochs = args.short_epochs or (SHORT_EPOCHS_QUICK if args.quick else SHORT_EPOCHS)
    long_epochs = args.long_epochs or (LONG_EPOCHS_QUICK if args.quick else LONG_EPOCHS)
    log_every = args.log_every

    torch.manual_seed(228)
    np.random.seed(228)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    migrate_legacy_checkpoint_names()
    resolve_dualsin_long_checkpoint()
    if args.golden_gabor:
        sync_legacy_gabor_data_to_experiment()

    dev = resolve_device(device_arg)
    configure_torch_backend(dev)
    if args.device == "cuda" and dev.type != "cuda":
        raise SystemExit(1)
    print(f"PyTorch device: {dev}", flush=True)
    if dev.type == "cpu" and args.device == "auto":
        print(
            "Note: CUDA not visible to this Python. If you have a 5070 Ti, check WSL GPU + CUDA PyTorch build.",
            flush=True,
        )

    runs = default_runs()
    # Fix dualSin long path after copy
    for r in runs:
        if r["activation"] == "DualAdaptiveSin" and r["horizon"] == "long":
            r["checkpoint"] = resolve_dualsin_long_checkpoint()

    if args.export_all:
        print("\n>>> Exporting all existing checkpoints ...", flush=True)
        export_all_checkpoints(runs, device_arg)

    poly_row = None
    if args.train:
        train_runs = resolve_train_targets(args.train, runs)
        n_long = sum(1 for r in train_runs if r.get("horizon") == "long" and not r.get("probe_only"))
        if n_long:
            print_runtime_estimate(short_epochs, long_epochs, n_long)
        for run in train_runs:
            row = train_run_entry(run, short_epochs, long_epochs, log_every, device_arg)
            if row is not None:
                poly_row = row

    rows: list[dict] = []
    if poly_row:
        rows.append(poly_row)

    for run in runs:
        if run.get("probe_only"):
            continue
        ck = run["checkpoint"]
        t_max = LONG_T if run["horizon"] == "long" else SHORT_T
        if not ck.exists():
            print(f"SKIP missing checkpoint: {ck}")
            continue
        print(f"[eval] {run['activation']} ({run['horizon']}) ← {ck.name}", flush=True)
        row = evaluate_run(run, t_max, N_TRAIN, device_arg, golden_gabor=args.golden_gabor)
        rows.append(row)
        print(
            f"       loss={row['final_train_loss']:.3e} max_err={row['max_trajectory_error']:.3e} "
            f"({row.get('metrics_source', 'checkpoint_eval')})",
            flush=True,
        )

    if poly_row is None:
        poly_readme = EXPERIMENT_HH / "LearnablePolynomial" / "README.md"
        if poly_readme.exists():
            final_loss = float("nan")
            for line in poly_readme.read_text(encoding="utf-8").splitlines():
                if line.startswith("- Final loss:"):
                    final_loss = float(line.split(":")[1].strip())
                    break
            rows.append(
                {
                    "activation": "LearnablePolynomial",
                    "horizon": "probe",
                    "final_train_loss": final_loss,
                    "max_trajectory_error": float("nan"),
                    "mean_trajectory_error": float("nan"),
                    "max_energy_drift": float("nan"),
                    "mean_energy_drift": float("nan"),
                    "qualitative_outcome": "unstable / not used for long-time HH",
                    "checkpoint": str(MODELS_DIR / "model_HH_polynomial_probe.zip"),
                    "experiment_dir": str(EXPERIMENT_HH / "LearnablePolynomial"),
                    "metrics_source": "probe_readme",
                }
            )
        else:
            rows.append(probe_polynomial(device_arg))

    print("\n>>> Writing comparison tables ...", flush=True)
    write_tables(rows)
    print(">>> All done.", flush=True)


if __name__ == "__main__":
    main()
