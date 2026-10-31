"""Scalar advection of a square wave, for chapter 1 of the walkthrough page.

u_t + a u_x = 0 on [0, 1], periodic, a = 1. The finite-volume scheme
updates cell averages with face fluxes a u*, where u* is the value taken at
the face: upwind (the cell the wave comes from), centred (mean of the two
neighbours) or MUSCL (limited linear reconstruction, upwinded).  Forward
Euler for first order, Heun for MUSCL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from euler1d.schemes.reconstruction.limiters import minmod, van_leer

SQUARE = (0.12, 0.38)   # u = 1 on this interval, 0 elsewhere (edges off the usual faces)
SPEED = 1.0
EXPLOSION = 10.0        # a run stops once max |u| exceeds this

PHI: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "fromm": lambda r: 0.5 * (1.0 + r),   # centred slope, no limiter
    "minmod": minmod,
    "van-leer": van_leer,
}


def exact_point(x: np.ndarray, t: float) -> np.ndarray:
    """Exact solution at points x: the square translated by a t (periodic)."""
    xi = np.mod(np.asarray(x, dtype=float) - SPEED * t, 1.0)
    return ((xi >= SQUARE[0]) & (xi < SQUARE[1])).astype(float)


def cell_centres(n_cells: int) -> np.ndarray:
    return (np.arange(n_cells) + 0.5) / n_cells


def exact_averages(n_cells: int, t: float) -> np.ndarray:
    """Exact cell averages: length of the cell covered by the square / dx."""
    dx = 1.0 / n_cells
    left = np.arange(n_cells) * dx
    shift = SPEED * t
    covered = np.zeros(n_cells)
    for k in (-1, 0, 1):   # periodic images of the translated square
        a, b = SQUARE[0] + shift % 1.0 + k, SQUARE[1] + shift % 1.0 + k
        covered += np.clip(np.minimum(left + dx, b) - np.maximum(left, a), 0.0, None)
    return covered / dx


def cut_cell(n_cells: int) -> int:
    """Cell holding the left edge of the square at t = 0; -1 if the edge is a face."""
    position = SQUARE[0] * n_cells
    return -1 if np.isclose(position, round(position)) else int(position)


def face_fluxes(u: np.ndarray, scheme: str, phi: Callable | None = None) -> np.ndarray:
    """Flux a u* at the right face of each cell (periodic)."""
    right = np.roll(u, -1)
    if scheme == "upwind":
        return SPEED * u
    if scheme == "centred":
        return SPEED * 0.5 * (u + right)
    if scheme == "muscl":
        d1 = right - u
        d0 = u - np.roll(u, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d1 != 0.0, d0 / np.where(d1 != 0.0, d1, 1.0), 0.0)
        slope = phi(r) * d1
        return SPEED * (u + 0.5 * slope)
    raise ValueError(f"Unknown scheme {scheme!r}")


@dataclass
class Run:
    u: np.ndarray       # last state computed (before an explosion, if any)
    steps_done: int
    exploded: bool


def advect(
    u0: np.ndarray, scheme: str, cfl: float, n_steps: int, phi: Callable | None = None
) -> Run:
    """Advance n_steps steps with dt = cfl dx / a; stop at an explosion."""
    nu = cfl  # a dt / dx

    def rate(v):
        F = face_fluxes(v, scheme, phi)
        return -nu * (F - np.roll(F, 1)) / SPEED

    u = np.asarray(u0, dtype=float).copy()
    for step in range(n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            if scheme == "muscl":
                u1 = u + rate(u)
                u_new = 0.5 * (u + u1 + rate(u1))
            else:
                u_new = u + rate(u)
        if not np.isfinite(u_new).all() or np.abs(u_new).max() > EXPLOSION:
            return Run(u, step, True)
        u = u_new
    return Run(u, n_steps, False)
