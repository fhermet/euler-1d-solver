"""Computations behind the finite-volume walkthrough page.

The page follows one time step of the solver on the Sod shock tube, cell by
cell: cell averages, reconstruction at the interfaces, local Riemann
problems, numerical fluxes, cell update and Runge-Kutta stages.  Every
quantity here comes from the solver's own functions (reconstruction, flux,
``_rk_step``, exact Riemann solver), so that what the page shows is what the
solver computes.  No Streamlit in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from euler1d.boundary import apply_boundary_conditions
from euler1d.config import GasProperties, SimulationConfig
from euler1d.physics import compute_flux, conservative_to_primitive, sound_speed
from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.flux.godunov import Godunov
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


def flat_cells(
    U: np.ndarray, scheme: NumericalScheme, gamma: float, dx: float, var: int
) -> np.ndarray:
    """Cells whose reconstruction of primitive variable *var* (0 rho, 1 u, 2 p)
    is flat while a neighbour differs: the slope was cancelled (limiter,
    extremum, discontinuity).  All False without reconstruction."""
    n = U.shape[1]
    if not isinstance(scheme, ReconstructedScheme):
        return np.zeros(n, dtype=bool)
    UL, UR = interface_states(U, scheme, gamma, dx)
    left_face = np.array(conservative_to_primitive(UR[:, :-1], gamma))[var]
    right_face = np.array(conservative_to_primitive(UL[:, 1:], gamma))[var]
    W = np.array(conservative_to_primitive(U, gamma))[var]
    W_ext = np.concatenate([W[:1], W, W[-1:]])
    neighbours = np.maximum(np.abs(W_ext[1:-1] - W_ext[:-2]), np.abs(W_ext[2:] - W_ext[1:-1]))
    scale = max(float(np.max(np.abs(W))), 1.0)
    tol = 1e-12 * scale
    return (np.abs(right_face - left_face) <= tol) & (neighbours > tol)


def default_interface(UL: np.ndarray, UR: np.ndarray) -> int:
    """Interface with the largest density jump."""
    return int(np.argmax(np.abs(UR[0] - UL[0])))


@dataclass
class FluxBreakdown:
    numerical: np.ndarray    # (3, N+1) flux of the scheme
    centred: np.ndarray      # 0.5 (F(UL) + F(UR))
    dissipation: np.ndarray  # centred - numerical
    godunov: np.ndarray      # exact Riemann (Godunov) flux of the same states


def flux_breakdown(
    UL: np.ndarray, UR: np.ndarray, scheme: NumericalScheme,
    gas: GasProperties, dx: float, dt: float,
) -> FluxBreakdown:
    numerical = scheme.compute_riemann_flux(UL, UR, gas, dx, dt)
    centred = 0.5 * (compute_flux(UL, gas.gamma) + compute_flux(UR, gas.gamma))
    godunov = Godunov().compute_riemann_flux(UL, UR, gas, dx, dt)
    return FluxBreakdown(numerical, centred, centred - numerical, godunov)


def stage_update(U: np.ndarray, F: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """One forward Euler update: U - (dt/dx) (F_{i+1/2} - F_{i-1/2})."""
    return U - (dt / dx) * (F[:, 1:] - F[:, :-1])


@dataclass
class Balance:
    change: np.ndarray    # (3,) change of the totals dx * sum(U)
    boundary: np.ndarray  # (3,) dt * (flux in at the left - flux out at the right)


def global_balance(
    U: np.ndarray, U_new: np.ndarray, F: np.ndarray, dt: float, dx: float
) -> Balance:
    return Balance(change=dx * (U_new - U).sum(axis=1), boundary=dt * (F[:, 0] - F[:, -1]))


def _rhs(scheme: NumericalScheme, config: SimulationConfig, dt: float):
    gas, dx = config.gas, config.mesh.dx
    return lambda v: _compute_rhs(v, scheme, gas, dx, dt, _BC, dt)


def rk_stages(
    U: np.ndarray, scheme: NumericalScheme, config: SimulationConfig,
    rk: str | None, dt: float,
) -> list[np.ndarray]:
    """States at which the Runge-Kutta method evaluates the spatial operator,
    followed by the final state of the step."""
    recorded: list[np.ndarray] = []
    rhs = _rhs(scheme, config, dt)

    def recording_rhs(v):
        recorded.append(v.copy())
        return rhs(v)

    final = _rk_step(U, select_time_integrator(scheme, rk), recording_rhs)
    return recorded + [final]


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
