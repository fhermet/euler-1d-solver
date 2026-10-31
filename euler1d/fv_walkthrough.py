"""Computations behind the finite-volume walkthrough page.

Chapter 2 of the walkthrough page ("et pour Euler ?") works on the Sod
shock tube: wave structure of the Riemann problem at the central face, the
wave model of each flux, and runs of first-order schemes.  Every quantity
comes from the solver's own functions (reconstruction, fluxes, _rk_step,
exact Riemann solver).  No Streamlit in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from euler1d.boundary import apply_boundary_conditions
from euler1d.config import GasProperties, SimulationConfig
from euler1d.physics import conservative_to_primitive, sound_speed
from euler1d.riemann import _vacuum_will_form, solve_riemann
from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.flux.hll import davis_speeds
from euler1d.schemes.flux.hllc import contact_speed
from euler1d.schemes.flux.roe import roe_eigenvalues
from euler1d.schemes.flux.rusanov import max_speed
from euler1d.schemes.reconstruction.base import ReconstructedScheme
from euler1d.solver import _compute_rhs, _rk_step, select_time_integrator
from euler1d.test_cases import sod_shock_tube

_BC = "transmissive"


def sod_setup(n_cells: int, cfl: float) -> SimulationConfig:
    """Sod shock tube on *n_cells* cells with the requested CFL number."""
    cfg = sod_shock_tube(n_cells)
    return replace(cfg, time=replace(cfg.time, cfl=cfl))


@dataclass
class TimeStep:
    dt: float
    cfl: float      # effective CFL: requested, capped by scheme.max_cfl
    s_max: float    # largest |u| + c over the cells
    i_max: int      # cell where it is reached


def time_step(U: np.ndarray, scheme: NumericalScheme, config: SimulationConfig) -> TimeStep:
    """Time step of the solver for state U (same rule as run_simulation)."""
    gamma = config.gas.gamma
    rho, u, p = conservative_to_primitive(U, gamma)
    speeds = np.abs(u) + sound_speed(rho, p, gamma)
    i_max = int(np.argmax(speeds))
    s_max = max(float(speeds[i_max]), 1e-14)
    cfl = min(config.time.cfl, scheme.max_cfl)
    return TimeStep(dt=cfl * config.mesh.dx / s_max, cfl=cfl, s_max=s_max, i_max=i_max)


def interface_states(
    U: np.ndarray, scheme: NumericalScheme, gamma: float, dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """Left and right states (3, N+1) at the interfaces, as the scheme builds them.

    Interface j lies between cells j-1 and j (j = 0 and N are the domain
    boundaries).  Without reconstruction they are the neighbouring averages.
    """
    g = scheme.n_ghost
    U_ext = apply_boundary_conditions(U, _BC, g)
    if isinstance(scheme, ReconstructedScheme):
        return scheme.reconstruct(U_ext, gamma, dx)
    n_intf = U.shape[1] + 1
    return U_ext[:, g - 1 : g - 1 + n_intf], U_ext[:, g : g + n_intf]


def _rhs(scheme: NumericalScheme, config: SimulationConfig, dt: float):
    gas, dx = config.gas, config.mesh.dx
    return lambda v: _compute_rhs(v, scheme, gas, dx, dt, _BC, dt)


@dataclass
class Advance:
    U: np.ndarray       # last physical state
    elapsed: float      # time advanced
    steps_done: int
    diverged: bool


def advance(
    U: np.ndarray, scheme: NumericalScheme, config: SimulationConfig,
    rk: str | None, n_steps: int,
) -> Advance:
    """Advance *n_steps* steps as run_simulation does; stop at the first
    non-finite state or non-positive density or pressure."""
    rk_used = select_time_integrator(scheme, rk)
    gamma = config.gas.gamma
    elapsed, done = 0.0, 0
    for _ in range(n_steps):
        dt = time_step(U, scheme, config).dt
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            U_new = _rk_step(U, rk_used, _rhs(scheme, config, dt))
            rho, _, p = conservative_to_primitive(U_new, gamma)
        if not (np.isfinite(U_new).all() and (rho > 0).all() and (p > 0).all()):
            return Advance(U, elapsed, done, True)
        U, elapsed, done = U_new, elapsed + dt, done + 1
    return Advance(U, elapsed, done, False)


@dataclass
class Wave:
    kind: str                    # "shock", "contact", "rarefaction" or "vacuum"
    speeds: tuple[float, ...]    # one speed; (head, tail) or (left edge, right edge)


def exact_waves(WL: tuple, WR: tuple, gamma: float) -> list[Wave]:
    """Waves of the exact Riemann solution between primitive states WL and WR,
    from left to right (Toro 2009, ch. 4)."""
    rhol, ul, pl = WL
    rhor, ur, pr = WR
    aL, aR = np.sqrt(gamma * pl / rhol), np.sqrt(gamma * pr / rhor)
    if _vacuum_will_form(rhol, ul, pl, rhor, ur, pr, gamma):
        tail_L = ul + 2.0 * aL / (gamma - 1.0)
        tail_R = ur - 2.0 * aR / (gamma - 1.0)
        return [Wave("rarefaction", (ul - aL, tail_L)), Wave("vacuum", (tail_L, tail_R)),
                Wave("rarefaction", (tail_R, ur + aR))]
    p_star, u_star, _, _ = solve_riemann(rhol, ul, pl, rhor, ur, pr, gamma)
    z = (gamma - 1.0) / (2.0 * gamma)
    shock_factor = lambda pk: np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / pk + z)
    if p_star > pl:
        left = Wave("shock", (ul - aL * shock_factor(pl),))
    else:
        left = Wave("rarefaction", (ul - aL, u_star - aL * (p_star / pl) ** z))
    if p_star > pr:
        right = Wave("shock", (ur + aR * shock_factor(pr),))
    else:
        right = Wave("rarefaction", (ur + aR, u_star + aR * (p_star / pr) ** z))
    return [left, Wave("contact", (u_star,)), right]


def model_waves(
    flux: str, UL: np.ndarray, UR: np.ndarray, gas: GasProperties
) -> list[float] | None:
    """Wave speeds that flux *flux* assumes between states UL and UR (shape (3,)).

    [] for Godunov (it uses the exact solution), None for AUSM+ and AUSM+-up,
    which split the flux without modelling waves.
    """
    UL, UR = np.asarray(UL, float).reshape(3, 1), np.asarray(UR, float).reshape(3, 1)
    if flux == "godunov":
        return []
    if flux in ("ausm+", "ausm+-up"):
        return None
    if flux == "rusanov":
        s = float(max_speed(UL, UR, gas.gamma)[0])
        return [-s, s]
    if flux in ("hll", "hllc"):
        SL, SR = davis_speeds(UL, UR, gas.gamma)
        if flux == "hll":
            return [float(SL[0]), float(SR[0])]
        S_star = contact_speed(UL, UR, gas.gamma, SL, SR)
        return [float(SL[0]), float(S_star[0]), float(SR[0])]
    if flux in ("roe", "roe-nc"):
        return [float(lam[0]) for lam in roe_eigenvalues(UL, UR, gas)]
    raise ValueError(f"No wave model for flux {flux!r}")


SOD_LEFT, SOD_RIGHT = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)


def contact_cells(rho: np.ndarray, x: np.ndarray, t: float, gamma: float = 1.4) -> int:
    """Number of cells across which the Sod contact is spread at time t.

    Cells within 0.1 of the exact contact (x = 0.5 + u* t) whose density lies
    strictly between the two star densities, trimmed by 10 % of the jump at
    each end.
    """
    _, u_star, rho_star_L, rho_star_R = solve_riemann(*SOD_LEFT, *SOD_RIGHT, gamma)
    band = 0.1 * (rho_star_L - rho_star_R)
    near = np.abs(x - (0.5 + u_star * t)) < 0.1
    inside = (rho > rho_star_R + band) & (rho < rho_star_L - band)
    return int(np.count_nonzero(near & inside))
