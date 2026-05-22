#!/usr/bin/env python3
"""Shared HH HNN utilities for training and metric evaluation.

Imported by hh_train_eval.py — not run directly.

Example:
    from hh_model import train_hh_model, evaluate_hh_model
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad

from hh_physics_utils import HH_exact, HHsolution, energy

dtype = torch.float


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """NumPy array on host; safe for CUDA tensors."""
    return t.detach().cpu().numpy()


def configure_torch_backend(dev: torch.device) -> None:
    """Lightweight backend tuning (safe for Hamiltonian autograd training)."""
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

# Matched to legacy_henon_heiles_train.py (commented short: 2e4, active long: 5e4).
# Not from project proposal — inherited from the Mattheakis / course HNN reference script.

ACTIVATION_NAMES = (
    "mySin",
    "AdaptiveSin",
    "DualAdaptiveSin",
    "LearnablePolynomial",
    "GaborActivation",
)


class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class AdaptiveSin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input):
        return torch.sin(self.a * input)


class DualAdaptiveSin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = torch.nn.Parameter(torch.tensor(0.5))
        self.a2 = torch.nn.Parameter(torch.tensor(1.5))
        self.w1 = torch.nn.Parameter(torch.tensor(0.5))
        self.w2 = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input):
        return self.w1 * torch.sin(self.a1 * input) + self.w2 * torch.sin(self.a2 * input)


class LearnablePolynomial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Parameter(torch.tensor(1.0))
        self.c2 = torch.nn.Parameter(torch.tensor(1e-4))
        self.c3 = torch.nn.Parameter(torch.tensor(1e-4))

    def forward(self, x):
        return self.c1 * x + self.c2 * torch.pow(x, 2) + self.c3 * torch.pow(x, 3)


class GaborActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.tensor(0.1))
        self.a = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.exp(-self.gamma * x * x) * torch.sin(self.a * x)


def make_activation(name: str) -> torch.nn.Module:
    table = {
        "mySin": mySin,
        "AdaptiveSin": AdaptiveSin,
        "DualAdaptiveSin": DualAdaptiveSin,
        "LearnablePolynomial": LearnablePolynomial,
        "GaborActivation": GaborActivation,
    }
    if name not in table:
        raise ValueError(f"Unknown activation: {name}")
    return table[name]()


def infer_activation_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = set(state_dict.keys())
    if any(k.startswith("actF.gamma") for k in keys):
        return "GaborActivation"
    if any(k.startswith("actF.c1") for k in keys):
        return "LearnablePolynomial"
    if any(k.startswith("actF.a1") for k in keys):
        return "DualAdaptiveSin"
    if any(k.startswith("actF.a") for k in keys):
        return "AdaptiveSin"
    return "mySin"


class odeNet_HH_MM(torch.nn.Module):
    def __init__(self, activation: str, D_hid: int = 80):
        super().__init__()
        self.actF = make_activation(activation)
        self.Lin_1 = torch.nn.Linear(1, D_hid)
        self.Lin_2 = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 4)

    def forward(self, t):
        l = self.Lin_1(t)
        h = self.actF(l)
        l = self.Lin_2(h)
        h = self.actF(l)
        r = self.Lin_out(h)
        xN = r[:, 0].reshape(-1, 1)
        yN = r[:, 1].reshape(-1, 1)
        pxN = r[:, 2].reshape(-1, 1)
        pyN = r[:, 3].reshape(-1, 1)
        return xN, yN, pxN, pyN


def dfx(x, f):
    return grad(
        [f],
        [x],
        grad_outputs=torch.ones_like(f),
        create_graph=True,
    )[0]


def perturbPoints(grid, t0, tf, sig=0.5):
    """Keep all tensors on the same device as grid (required for CUDA training)."""
    delta_t = grid[1] - grid[0]
    noise = delta_t * torch.randn_like(grid) * sig
    t = grid + noise
    if t.shape[0] > 2:
        t[2, 0] = -1.0
    t = torch.where(t < t0, torch.full_like(t, t0), t)
    t = torch.where(t > tf, 2.0 * tf - t, t)
    t.requires_grad = False
    return t


def parametricSolutions(t, nn, X0):
    t0, x0, y0, px0, py0, _ = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    N1, N2, N3, N4 = nn(t)
    t0_t = torch.tensor(t0, device=t.device, dtype=dtype)
    dt = t - t0_t
    f = 1 - torch.exp(-dt)
    return x0 + f * N1, y0 + f * N2, px0 + f * N3, py0 + f * N4


def hamEqs_Loss(t, x, y, px, py, lam):
    xd, yd, pxd, pyd = dfx(t, x), dfx(t, y), dfx(t, px), dfx(t, py)
    fx = xd - px
    fy = yd - py
    fpx = pxd + x + 2.0 * lam * x * y
    fpy = pyd + y + lam * (x.pow(2) - y.pow(2))
    return (fx.pow(2)).mean() + (fy.pow(2)).mean() + (fpx.pow(2)).mean() + (fpy.pow(2)).mean()


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick training device. Use --device cuda|cpu in hh_train_eval.py."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "In WSL: install NVIDIA driver on Windows, enable WSL2 GPU, "
                "and install PyTorch with CUDA (pip install torch --index-url https://download.pytorch.org/whl/cu124)."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hamiltonian_torch(x, y, px, py, lam):
    V = 0.5 * (x**2 + y**2) + lam * (x**2 * y - y**3 / 3)
    K = 0.5 * (px**2 + py**2)
    return K + V


def _format_eta(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def train_hh_model(
    activation: str,
    X0: List[float],
    t_max: float,
    n_train: int,
    neurons: int,
    epochs: int,
    lr: float,
    checkpoint: Path,
    load_weights: bool = False,
    min_loss: float = 1e-8,
    perturb_sigma: float = 0.3,
    log_every: int = 500,
    stage_label: str = "",
    device: str | None = None,
) -> Tuple[torch.nn.Module, List[float], float]:
    dev = resolve_device(device)
    configure_torch_backend(dev)
    fc0 = odeNet_HH_MM(activation, D_hid=neurons).to(dev)
    fc1 = copy.deepcopy(fc0)
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=[0.999, 0.9999])
    loss_history: List[float] = []

    t0 = X0[0]
    x0, y0, px0, py0, lam = X0[1], X0[2], X0[3], X0[4], X0[5]
    ham0 = hamiltonian_torch(
        torch.tensor(x0, dtype=dtype, device=dev),
        torch.tensor(y0, dtype=dtype, device=dev),
        torch.tensor(px0, dtype=dtype, device=dev),
        torch.tensor(py0, dtype=dtype, device=dev),
        lam,
    )
    grid = torch.linspace(t0, t_max, n_train, device=dev).reshape(-1, 1)
    Llim = 1.0

    if checkpoint.exists() and load_weights:
        ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
        fc0.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    label = stage_label or f"{activation} train"
    header = (
        f"\n[{label}] device={dev} epochs={epochs}, points={n_train}, t_max={t_max:.2f}, "
        f"neurons={neurons}, lr={lr}, ckpt={checkpoint.name}"
    )
    print(header, flush=True)

    t_start = time.time()
    pbar = None
    epoch_iter: range | object = range(epochs)
    if tqdm is not None:
        pbar = tqdm(
            range(epochs),
            desc=label,
            unit="epoch",
            file=sys.stdout,
            mininterval=1.0,
        )
        epoch_iter = pbar

    for tt in epoch_iter:
        t = perturbPoints(grid, t0, t_max, sig=perturb_sigma * t_max)
        t.requires_grad = True
        x, y, px, py = parametricSolutions(t, fc0, X0)
        Ltot = hamEqs_Loss(t, x, y, px, py, lam)
        ham = hamiltonian_torch(x, y, px, py, lam)
        Ltot = Ltot + 0.5 * ((ham - ham0).pow(2)).mean()

        if not torch.isfinite(Ltot):
            loss_history.append(float("nan"))
            print(f"\n[{label}] stopped at epoch {tt}: non-finite loss", flush=True)
            break

        Ltot.backward(retain_graph=False)
        optimizer.step()
        loss = float(Ltot.item())
        optimizer.zero_grad()
        loss_history.append(loss)

        if tqdm is None and log_every > 0 and (tt == 0 or (tt + 1) % log_every == 0 or tt + 1 == epochs):
            elapsed = time.time() - t_start
            done = tt + 1
            rate = done / max(elapsed, 1e-9)
            eta = (epochs - done) / max(rate, 1e-9)
            pct = 100.0 * done / epochs
            print(
                f"[{label}] {done}/{epochs} ({pct:5.1f}%) "
                f"loss={loss:.4e} elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}",
                flush=True,
            )
        elif pbar is not None:
            pbar.set_postfix(loss=f"{loss:.2e}", refresh=False)

        if tt > 0.8 * epochs and Ltot < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = Ltot
        if Ltot < min_loss:
            fc1 = copy.deepcopy(fc0)
            print(f"\n[{label}] early stop at epoch {tt}, loss={loss:.4e}", flush=True)
            break

    run_time = time.time() - t_start
    final = loss_history[-1] if loss_history else float("nan")
    print(
        f"[{label}] done in {_format_eta(run_time)} | final_loss={final:.4e} | "
        f"epochs_run={len(loss_history)}",
        flush=True,
    )
    torch.save(
        {
            "epoch": len(loss_history) - 1,
            "model_state_dict": fc1.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_history[-1] if loss_history else float("nan"),
            "activation": activation,
        },
        checkpoint,
    )
    return fc1, loss_history, run_time


def evaluate_hh_model(
    model: torch.nn.Module,
    X0: List[float],
    t_max: float,
    n_test: int,
    device: str | None = None,
) -> Dict[str, float]:
    dev = resolve_device(device)
    model = model.to(dev)
    t0, x0, y0, px0, py0, lam = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    t_test = torch.linspace(t0, t_max, n_test, device=dev).reshape(-1, 1)
    t_test.requires_grad = True
    x, y, px, py = parametricSolutions(t_test, model, X0)
    x_np = tensor_to_numpy(x)[:, 0]
    y_np = tensor_to_numpy(y)[:, 0]
    px_np = tensor_to_numpy(px)[:, 0]
    py_np = tensor_to_numpy(py)[:, 0]
    E_net = energy(x_np, y_np, px_np, py_np, lam)

    t_num = np.linspace(t0, t_max, n_test)
    E0, _ = HH_exact(n_test, x0, y0, px0, py0, lam)
    x_num, y_num, px_num, py_num = HHsolution(n_test, t_num, x0, y0, px0, py0, lam)

    dx = x_num - x_np
    dy = y_num - y_np
    dpx = px_num - px_np
    dpy = py_num - py_np
    state_err = np.sqrt(dx**2 + dy**2 + dpx**2 + dpy**2)
    energy_drift = np.abs(E_net - E0)

    return {
        "max_trajectory_error": float(np.max(state_err)),
        "mean_trajectory_error": float(np.mean(state_err)),
        "max_energy_drift": float(np.max(energy_drift)),
        "mean_energy_drift": float(np.mean(energy_drift)),
        "E0": float(E0),
    }


def load_model_from_checkpoint(
    checkpoint: Path, neurons: int = 80, device: str | None = None
) -> Tuple[torch.nn.Module, Optional[float]]:
    dev = resolve_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    activation = ckpt.get("activation") or infer_activation_from_state_dict(ckpt["model_state_dict"])
    model = odeNet_HH_MM(activation, D_hid=neurons).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    final_loss = ckpt.get("loss")
    if hasattr(final_loss, "item"):
        final_loss = float(final_loss.item())
    elif final_loss is not None:
        final_loss = float(final_loss)
    return model, final_loss
