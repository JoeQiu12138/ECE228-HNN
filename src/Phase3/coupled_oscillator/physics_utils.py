#!/usr/bin/env python3
"""Ground truth for coupled nonlinear oscillators (Phase 3)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint


def energy(x1, x2, p1, p2, beta: float = 0.5) -> np.ndarray:
    x1 = np.asarray(x1).reshape(-1)
    x2 = np.asarray(x2).reshape(-1)
    p1 = np.asarray(p1).reshape(-1)
    p2 = np.asarray(p2).reshape(-1)
    E = 0.5 * (p1**2 + p2**2 + x1**2 + x2**2) + beta * (x1**2 * x2**2)
    return E.reshape(-1)


def _rhs(u, t, beta: float):
    x1, x2, p1, p2 = u
    return [
        p1,
        p2,
        -x1 - 2.0 * beta * x1 * (x2**2),
        -x2 - 2.0 * beta * x2 * (x1**2),
    ]


def coupled_solution(N: int, t, x10, x20, p10, p20, beta: float = 0.5):
    u0 = [x10, x20, p10, p20]
    sol = odeint(_rhs, u0, t, args=(beta,))
    return sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]


def coupled_exact(N: int, x10, x20, p10, p20, beta: float = 0.5):
    E0 = float(energy(np.array([x10]), np.array([x20]), np.array([p10]), np.array([p20]), beta)[0])
    return E0, E0 * np.ones(N)


def symplectic_euler(Ns: int, x10, x20, p10, p20, t0: float, t_max: float, beta: float = 0.5):
    t_s = np.linspace(t0, t_max, Ns + 1)
    dt = t_max / Ns
    x1 = np.zeros(Ns + 1)
    x2 = np.zeros(Ns + 1)
    p1 = np.zeros(Ns + 1)
    p2 = np.zeros(Ns + 1)
    x1[0], x2[0], p1[0], p2[0] = x10, x20, p10, p20
    for n in range(Ns):
        x1[n + 1] = x1[n] + dt * p1[n]
        x2[n + 1] = x2[n] + dt * p2[n]
        p1[n + 1] = p1[n] - dt * (x1[n + 1] + 2.0 * beta * x1[n + 1] * (x2[n + 1] ** 2))
        p2[n + 1] = p2[n] - dt * (x2[n + 1] + 2.0 * beta * x2[n + 1] * (x1[n + 1] ** 2))
    E_s = energy(x1, x2, p1, p2, beta)
    return E_s, x1, x2, p1, p2, t_s
