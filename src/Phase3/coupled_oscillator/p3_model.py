#!/usr/bin/env python3
"""Coupled-oscillator HNN model (Phase 3). Reuses activations from Phase 2 hh_model."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad

ROOT = Path(__file__).resolve().parent
PHASE2 = ROOT.parent.parent / "Phase2" / "HHsystem"
if str(PHASE2) not in sys.path:
    sys.path.insert(0, str(PHASE2))

from hh_model import (  # noqa: E402
    configure_torch_backend,
    dfx,
    infer_activation_from_state_dict,
    make_activation,
    perturbPoints,
    resolve_device,
)
from physics_utils import coupled_exact, coupled_solution, energy

dtype = torch.float


class odeNet_coupled(torch.nn.Module):
    def __init__(self, activation: str, D_hid: int = 80):
        super().__init__()
        self.actF = make_activation(activation)
        self.Lin_1 = torch.nn.Linear(1, D_hid)
        self.Lin_2 = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 4)

    def forward(self, t):
        h = self.actF(self.Lin_1(t))
        h = self.actF(self.Lin_2(h))
        r = self.Lin_out(h)
        return (
            r[:, 0].reshape(-1, 1),
            r[:, 1].reshape(-1, 1),
            r[:, 2].reshape(-1, 1),
            r[:, 3].reshape(-1, 1),
        )


def parametric_solutions(t, nn: odeNet_coupled, X0: List[float]):
    t0, x10, x20, p10, p20, beta = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    N1, N2, N3, N4 = nn(t)
    dt = t - torch.tensor(t0, device=t.device, dtype=dtype)
    f = 1.0 - torch.exp(-dt)
    return (
        x10 + f * N1,
        x20 + f * N2,
        p10 + f * N3,
        p20 + f * N4,
    )


def hamiltonian_torch(x1, x2, p1, p2, beta: float):
    V = 0.5 * (x1**2 + x2**2) + beta * (x1**2 * x2**2)
    K = 0.5 * (p1**2 + p2**2)
    return K + V


def ham_eqs_loss(t, x1, x2, p1, p2, beta: float):
    x1d, x2d, p1d, p2d = dfx(t, x1), dfx(t, x2), dfx(t, p1), dfx(t, p2)
    fx1 = x1d - p1
    fx2 = x2d - p2
    fp1 = p1d + x1 + 2.0 * beta * x1 * (x2**2)
    fp2 = p2d + x2 + 2.0 * beta * x2 * (x1**2)
    return (fx1.pow(2)).mean() + (fx2.pow(2)).mean() + (fp1.pow(2)).mean() + (fp2.pow(2)).mean()


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


def train_coupled_model(
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
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fc0 = odeNet_coupled(activation, D_hid=neurons).to(dev)
    fc1 = copy.deepcopy(fc0)
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=[0.999, 0.9999])
    loss_history: List[float] = []

    t0 = X0[0]
    x10, x20, p10, p20, beta = X0[1], X0[2], X0[3], X0[4], X0[5]
    ham0 = hamiltonian_torch(
        torch.tensor(x10, dtype=dtype, device=dev),
        torch.tensor(x20, dtype=dtype, device=dev),
        torch.tensor(p10, dtype=dtype, device=dev),
        torch.tensor(p20, dtype=dtype, device=dev),
        beta,
    )
    grid = torch.linspace(t0, t_max, n_train, device=dev).reshape(-1, 1)
    Llim = 1.0

    if checkpoint.exists() and load_weights:
        ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
        fc0.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    label = stage_label or f"{activation} train"
    print(
        f"\n[{label}] device={dev} epochs={epochs} points={n_train} t_max={t_max:.2f} "
        f"ckpt={checkpoint.name}",
        flush=True,
    )
    t_start = time.time()

    for tt in range(epochs):
        t = perturbPoints(grid, t0, t_max, sig=perturb_sigma * t_max)
        t.requires_grad = True
        x1, x2, p1, p2 = parametric_solutions(t, fc0, X0)
        Ltot = ham_eqs_loss(t, x1, x2, p1, p2, beta)
        ham = hamiltonian_torch(x1, x2, p1, p2, beta)
        Ltot = Ltot + 0.5 * ((ham - ham0).pow(2)).mean()

        if not torch.isfinite(Ltot):
            loss_history.append(float("nan"))
            print(f"\n[{label}] stopped at epoch {tt}: non-finite loss", flush=True)
            break

        Ltot.backward(retain_graph=False)
        optimizer.step()
        loss_history.append(float(Ltot.item()))
        optimizer.zero_grad()

        if log_every > 0 and (tt == 0 or (tt + 1) % log_every == 0 or tt + 1 == epochs):
            elapsed = time.time() - t_start
            done = tt + 1
            rate = done / max(elapsed, 1e-9)
            eta = (epochs - done) / max(rate, 1e-9)
            print(
                f"[{label}] {done}/{epochs} loss={loss_history[-1]:.4e} "
                f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}",
                flush=True,
            )

        if tt > 0.8 * epochs and Ltot < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = Ltot
        if Ltot < min_loss:
            fc1 = copy.deepcopy(fc0)
            print(f"\n[{label}] early stop epoch {tt}", flush=True)
            break

    run_time = time.time() - t_start
    torch.save(
        {
            "epoch": len(loss_history) - 1,
            "model_state_dict": fc1.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_history[-1] if loss_history else float("nan"),
            "activation": activation,
            "system": "coupled_oscillator",
        },
        checkpoint,
    )
    return fc1, loss_history, run_time


def evaluate_coupled_model(
    model: torch.nn.Module,
    X0: List[float],
    t_max: float,
    n_test: int,
    device: str | None = None,
) -> Dict[str, float]:
    dev = resolve_device(device)
    model = model.to(dev)
    t0, x10, x20, p10, p20, beta = X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]
    t_test = torch.linspace(t0, t_max, n_test, device=dev).reshape(-1, 1)
    t_test.requires_grad = True
    x1, x2, p1, p2 = parametric_solutions(t_test, model, X0)
    x1_np = x1.detach().cpu().numpy()[:, 0]
    x2_np = x2.detach().cpu().numpy()[:, 0]
    p1_np = p1.detach().cpu().numpy()[:, 0]
    p2_np = p2.detach().cpu().numpy()[:, 0]
    E_net = energy(x1_np, x2_np, p1_np, p2_np, beta)

    t_num = np.linspace(t0, t_max, n_test)
    E0, _ = coupled_exact(n_test, x10, x20, p10, p20, beta)
    x1_num, x2_num, p1_num, p2_num = coupled_solution(n_test, t_num, x10, x20, p10, p20, beta)

    err = np.sqrt(
        (x1_num - x1_np) ** 2
        + (x2_num - x2_np) ** 2
        + (p1_num - p1_np) ** 2
        + (p2_num - p2_np) ** 2
    )
    drift = np.abs(E_net - E0)
    return {
        "max_trajectory_error": float(np.max(err)),
        "mean_trajectory_error": float(np.mean(err)),
        "max_energy_drift": float(np.max(drift)),
        "mean_energy_drift": float(np.mean(drift)),
        "E0": float(E0),
    }


def load_model_from_checkpoint(
    checkpoint: Path, neurons: int = 80, device: str | None = None
) -> Tuple[torch.nn.Module, Optional[float]]:
    dev = resolve_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    activation = ckpt.get("activation") or infer_activation_from_state_dict(ckpt["model_state_dict"])
    model = odeNet_coupled(activation, D_hid=neurons).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    final_loss = ckpt.get("loss")
    if hasattr(final_loss, "item"):
        final_loss = float(final_loss.item())
    elif final_loss is not None:
        final_loss = float(final_loss)
    return model, final_loss
