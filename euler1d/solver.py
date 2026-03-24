"""Time-stepping loop for the Euler 1D solver.

This module implements the **method of lines** approach for solving the 1D Euler
equations: the spatial discretisation (boundary conditions, reconstruction, flux
computation) produces a right-hand side operator L(U), and an explicit
Runge-Kutta integrator advances the solution in time.

Five RK integrators are available (RK1 through RK5), ranging from first-order
Euler explicit to the sixth-stage Dormand-Prince method of order 5.  The
integrator is chosen automatically to match the spatial order of the numerical
scheme, ensuring space-time consistency (see docs/06_time_integration.md, §6.7).

The time step is adaptive: it is recomputed every iteration from the CFL
condition Δt = CFL × Δx / max(|u| + c), guaranteeing that the fastest wave
does not cross more than one cell per step (§6.1).

See Also
--------
docs/06_time_integration.md : Full theoretical description (CFL, RK methods,
    SSP property, Butcher tableaux, space-time consistency).
euler1d.schemes.base.NumericalScheme : Spatial operator interface.
euler1d.boundary : Ghost-cell boundary conditions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from euler1d.boundary import apply_boundary_conditions
from euler1d.config import RiemannProblem, SimulationConfig, SmoothProblem
from euler1d.physics import max_wave_speed, primitive_to_conservative
from euler1d.schemes.base import NumericalScheme


@dataclass
class SolverResult:
    """Container for the output of a simulation run.

    Stores the final solution together with intermediate snapshots (for
    animation or time-history analysis) and timing metadata.

    Attributes
    ----------
    x : np.ndarray
        Cell-centre coordinates, shape ``(N,)``.
    times : list[float]
        Physical times at which snapshots were recorded.
    snapshots : list[np.ndarray]
        Conservative-variable arrays at each snapshot time, each of shape
        ``(3, N)`` with rows (rho, rho*u, E).  The last snapshot always
        corresponds to ``t_final``.
    scheme_name : str
        Human-readable name of the numerical scheme used.
    config : SimulationConfig or None
        The full simulation configuration (mesh, gas, time, problem).  Kept
        for post-processing convenience (e.g. recomputing the exact solution).
    wall_time : float
        Wall-clock time of the simulation loop in seconds, useful for
        benchmarking scheme performance.
    """

    x: np.ndarray
    times: list[float] = field(default_factory=list)
    snapshots: list[np.ndarray] = field(default_factory=list)
    scheme_name: str = ""
    config: SimulationConfig | None = None
    wall_time: float = 0.0


def _compute_rhs(
    U: np.ndarray,
    scheme: NumericalScheme,
    gas,
    dx: float,
    dt: float,
    bc_type: str,
) -> np.ndarray:
    """Compute the scaled right-hand side Δt·L(U) of the semi-discrete system.

    This function encapsulates one complete spatial evaluation:

    1. Apply boundary conditions (ghost cells) to *U* according to *bc_type*.
    2. Call ``scheme.compute_fluxes`` on the extended array to obtain the
       numerical fluxes F̂_{i+1/2} at the N+1 cell interfaces.
    3. Return the flux-difference vector scaled by Δt/Δx:

       Δt·L(U)_i = -(Δt/Δx) · (F̂_{i+1/2} − F̂_{i-1/2})

    The result is a conservative-variable increment of shape ``(3, N)`` that
    can be directly added to U in the Runge-Kutta stages.

    Parameters
    ----------
    U : np.ndarray
        Conservative variables, shape ``(3, N)``.
    scheme : NumericalScheme
        Numerical scheme providing ``compute_fluxes`` and ``n_ghost``.
    gas : GasConstants
        Gas model (gamma).
    dx : float
        Cell width.
    dt : float
        Current time step.
    bc_type : str
        Boundary condition type (``"transmissive"``, ``"reflective"``,
        ``"periodic"``).

    Returns
    -------
    np.ndarray
        Δt·L(U), shape ``(3, N)``.

    See Also
    --------
    docs/06_time_integration.md §6.1, docs/03_finite_volume.md §3.3.
    """
    U_ext = apply_boundary_conditions(U, bc_type, n_ghost=scheme.n_ghost)
    F = scheme.compute_fluxes(U_ext, gas, dx, dt)
    return -(dt / dx) * (F[:, 1:] - F[:, :-1])


TIME_INTEGRATORS = ("RK1", "RK2", "RK3", "RK4", "RK5")


def run_simulation(
    config: SimulationConfig,
    scheme: NumericalScheme,
    n_snapshots: int = 1,
    progress_callback: Callable[[float], None] | None = None,
    time_integrator: str | None = None,
) -> SolverResult:
    """Run the Euler 1D simulation using the method of lines.

    This is the main entry point of the solver.  It initialises the
    conservative-variable array from the problem definition, selects the
    appropriate Runge-Kutta integrator, and advances the solution from
    ``t = 0`` to ``t = t_final`` with an adaptive time step governed by
    the CFL condition (see docs/06_time_integration.md, §6.1).

    **Time-integrator selection logic** (§6.7):

    1. If ``scheme.time_integral_included`` is True (e.g. Lax-Wendroff), the
       integrator is forced to RK1 to avoid double time stepping.
    2. If the caller supplies *time_integrator*, that value is used.
    3. Otherwise ``scheme.default_time_integrator`` is used, which maps the
       spatial order to the matching RK method (order 1 → RK1, 2 → RK2,
       3 → SSP-RK3, 4 → RK4, 5 → Dormand-Prince RK5).

    At each iteration the loop:

    * Computes the adaptive Δt from the maximum wave speed.
    * Adjusts Δt so as not to overshoot ``t_final``.
    * Performs the RK stages (each stage calls ``_compute_rhs``).
    * Records snapshots at evenly spaced times (plus the final time).

    Parameters
    ----------
    config : SimulationConfig
        Full problem specification (mesh, gas, time control, problem
        definition, boundary condition type).
    scheme : NumericalScheme
        Spatial discretisation (flux + optional reconstruction).
    n_snapshots : int, optional
        Number of intermediate snapshots to record in addition to the
        final state.  Set to 0 for final-state-only output.  Default is 1.
    progress_callback : callable or None, optional
        If provided, called after every time step with a float in [0, 1]
        representing the fraction of ``t_final`` reached.  Useful for
        progress bars in the Streamlit UI.
    time_integrator : str or None, optional
        Explicit RK method to use (``"RK1"`` .. ``"RK5"``).  If None the
        scheme's default integrator is used.

    Returns
    -------
    SolverResult
        Solution snapshots, timing information, and configuration metadata.

    See Also
    --------
    docs/06_time_integration.md : §6.1 (CFL), §6.2-6.6 (RK methods),
        §6.7 (space-time consistency).
    """
    mesh = config.mesh
    gas = config.gas
    tc = config.time
    prob = config.problem

    dx = mesh.dx
    x = np.linspace(mesh.x_min + 0.5 * dx, mesh.x_max - 0.5 * dx, mesh.n_cells)

    # Initial conditions
    if isinstance(prob, SmoothProblem):
        rho, u, p = prob.init_fn(x, gas.gamma)
    else:
        rho = np.where(x < prob.x_discontinuity, prob.left.rho, prob.right.rho)
        u = np.where(x < prob.x_discontinuity, prob.left.u, prob.right.u)
        p = np.where(x < prob.x_discontinuity, prob.left.p, prob.right.p)
    U = primitive_to_conservative(rho, u, p, gas.gamma)

    t = 0.0
    t_final = tc.t_final

    # Schemes with built-in time integration (e.g. Lax-Wendroff) must use RK1
    if scheme.time_integral_included:
        rk = "RK1"
    elif time_integrator is None:
        rk = scheme.default_time_integrator
    else:
        rk = time_integrator

    # Snapshot times
    if n_snapshots > 0:
        snapshot_times = set(
            t_final * (i + 1) / n_snapshots for i in range(n_snapshots)
        )
    else:
        snapshot_times = set()
    snapshot_times.add(t_final)

    result = SolverResult(x=x, scheme_name=scheme.name, config=config)

    t_start = time.perf_counter()

    while t < t_final:
        # Detect divergence (NaN in solution)
        if not np.isfinite(U).all():
            import warnings
            warnings.warn(
                f"Solution diverged at t={t:.6e} (NaN/Inf detected). "
                f"Saving last valid state.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # Compute time step
        s_max = max_wave_speed(U, gas.gamma)
        if s_max < 1e-14:
            s_max = 1e-14
        if tc.dt_fixed is not None:
            dt = tc.dt_fixed
        else:
            dt = tc.cfl * dx / s_max

        # Don't overshoot
        if t + dt > t_final:
            dt = t_final - t

        k1 = _compute_rhs(U, scheme, gas, dx, dt, config.bc_type)
        if rk == "RK1":
            U = U + k1
        elif rk == "RK2":
            k2 = _compute_rhs(U + k1, scheme, gas, dx, dt, config.bc_type)
            U = U + 0.5 * (k1 + k2)
        elif rk == "RK3":
            # SSP-RK3 (Shu-Osher)
            U1 = U + k1
            k2 = _compute_rhs(U1, scheme, gas, dx, dt, config.bc_type)
            U2 = 0.75 * U + 0.25 * (U1 + k2)
            k3 = _compute_rhs(U2, scheme, gas, dx, dt, config.bc_type)
            U = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + k3)
        elif rk == "RK4":
            k2 = _compute_rhs(U + 0.5 * k1, scheme, gas, dx, dt, config.bc_type)
            k3 = _compute_rhs(U + 0.5 * k2, scheme, gas, dx, dt, config.bc_type)
            k4 = _compute_rhs(U + k3, scheme, gas, dx, dt, config.bc_type)
            U = U + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        else:  # RK5 — Dormand-Prince (5th order, 6 stages)
            rhs = lambda v: _compute_rhs(v, scheme, gas, dx, dt, config.bc_type)
            k2 = rhs(U + (1.0/5.0) * k1)
            k3 = rhs(U + (3.0/40.0) * k1 + (9.0/40.0) * k2)
            k4 = rhs(U + (44.0/45.0) * k1 - (56.0/15.0) * k2 + (32.0/9.0) * k3)
            k5 = rhs(U + (19372.0/6561.0) * k1 - (25360.0/2187.0) * k2
                     + (64448.0/6561.0) * k3 - (212.0/729.0) * k4)
            k6 = rhs(U + (9017.0/3168.0) * k1 - (355.0/33.0) * k2
                     + (46732.0/5247.0) * k3 + (49.0/176.0) * k4
                     - (5103.0/18656.0) * k5)
            U = U + (35.0/384.0) * k1 + (500.0/1113.0) * k3 \
                  + (125.0/192.0) * k4 - (2187.0/6784.0) * k5 \
                  + (11.0/84.0) * k6

        # Positivity floor: prevent negative density/pressure from crashing
        # the simulation (can happen with schemes lacking entropy fix).
        _FLOOR = 1e-10
        U[0] = np.maximum(U[0], _FLOOR)  # rho >= floor
        ke = 0.5 * U[1]**2 / U[0]        # kinetic energy
        internal = U[2] - ke              # internal energy = E - 0.5*rho*u^2
        U[2] = np.where(internal < _FLOOR, ke + _FLOOR, U[2])

        t += dt

        # Check for snapshots
        to_remove = set()
        for ts in snapshot_times:
            if t >= ts - 1e-14:
                result.times.append(t)
                result.snapshots.append(U.copy())
                to_remove.add(ts)
        snapshot_times -= to_remove

        if progress_callback is not None:
            progress_callback(min(t / t_final, 1.0))

    # If no snapshots were recorded (e.g. early divergence), save current state
    if not result.snapshots:
        result.times.append(t)
        result.snapshots.append(U.copy())

    result.wall_time = time.perf_counter() - t_start
    return result
