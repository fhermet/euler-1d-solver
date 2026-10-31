"""Time-stepping loop for the Euler 1D solver.

This module implements the **method of lines** approach for solving the 1D Euler
equations: the spatial discretisation (boundary conditions, reconstruction, flux
computation) produces a right-hand side operator L(U), and an explicit
Runge-Kutta integrator advances the solution in time.

Five RK integrators are available (RK1 through RK5), ranging from first-order
Euler explicit to the sixth-stage Dormand-Prince method of order 5.  By
default the integrator follows the spatial order up to 3 (RK1, RK2, SSP-RK3);
higher orders also use SSP-RK3 for stability on discontinuities (see
docs/06_time_integration.md, §6.7).

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
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from euler1d.boundary import apply_boundary_conditions
from euler1d.config import RiemannProblem, SimulationConfig, SmoothProblem
from euler1d.physics import (
    cell_average,
    conservative_to_primitive,
    max_wave_speed,
    primitive_to_conservative,
)
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
    status : str
        ``"ok"`` if ``t_final`` was reached, ``"diverged"`` if a non-finite
        state or a non-positive density or pressure appeared.  On divergence
        the last snapshot is the last physical state, recorded at the time it
        was reached.
    floor_events : int
        Number of cell corrections made by the positivity floor.  Non-zero
        means the scheme alone produced a non-physical state.
    floor_energy : float
        Total energy (integrated over the domain) injected by the floor.
    cfl : float
        CFL number actually used: the requested one, capped by the scheme's
        ``max_cfl``.
    """

    x: np.ndarray
    times: list[float] = field(default_factory=list)
    snapshots: list[np.ndarray] = field(default_factory=list)
    scheme_name: str = ""
    config: SimulationConfig | None = None
    wall_time: float = 0.0
    status: str = "ok"
    floor_events: int = 0
    floor_energy: float = 0.0
    cfl: float = 0.0


def _compute_rhs(
    U: np.ndarray,
    scheme: NumericalScheme,
    gas,
    dx: float,
    dt: float,
    bc_type: str,
    dt_flux: float | None = None,
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
    dt_flux : float or None
        Time step handed to the numerical flux, if different from *dt*.
        Defaults to *dt*.

    Returns
    -------
    np.ndarray
        Δt·L(U), shape ``(3, N)``.

    See Also
    --------
    docs/06_time_integration.md §6.1, docs/03_finite_volume.md §3.3.
    """
    U_ext = apply_boundary_conditions(U, bc_type, n_ghost=scheme.n_ghost)
    F = scheme.compute_fluxes(U_ext, gas, dx, dt if dt_flux is None else dt_flux)
    return -(dt / dx) * (F[:, 1:] - F[:, :-1])


TIME_INTEGRATORS = ("RK1", "RK2", "RK3", "RK4", "RK5")


def select_time_integrator(scheme: NumericalScheme, requested: str | None) -> str:
    """Runge-Kutta method actually used for *scheme*.

    Schemes whose flux already embeds the time integration (Lax-Wendroff)
    always use RK1, whatever is requested; otherwise the requested method,
    or the scheme's default when None ("Auto").
    """
    if scheme.time_integral_included:
        return "RK1"
    return requested or scheme.default_time_integrator


def initial_state(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres and initial cell averages U(3, N).

    Smooth problems are averaged over each cell by Gauss quadrature.  For a
    Riemann problem the cell that contains the discontinuity mixes the two
    states in proportion to its length on each side.
    """
    mesh, gamma, prob = config.mesh, config.gas.gamma, config.problem
    dx = mesh.dx
    x = np.linspace(mesh.x_min + 0.5 * dx, mesh.x_max - 0.5 * dx, mesh.n_cells)
    if isinstance(prob, SmoothProblem):
        return x, cell_average(lambda s: prob.init_fn(s, gamma), x, dx, gamma)
    UL = primitive_to_conservative(prob.left.rho, prob.left.u, prob.left.p, gamma)
    UR = primitive_to_conservative(prob.right.rho, prob.right.u, prob.right.p, gamma)
    left_fraction = np.clip((prob.x_discontinuity - (x - 0.5 * dx)) / dx, 0.0, 1.0)
    return x, left_fraction * UL[:, None] + (1.0 - left_fraction) * UR[:, None]

# Positivity floor on density and internal energy (see run_simulation).
_FLOOR = 1e-10


def _rk_step(
    U: np.ndarray, rk: str, rhs: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Advance U by one time step; *rhs* returns Δt·L(U)."""
    k1 = rhs(U)
    if rk == "RK1":
        return U + k1
    if rk == "RK2":
        k2 = rhs(U + k1)
        return U + 0.5 * (k1 + k2)
    if rk == "RK3":
        # SSP-RK3 (Shu-Osher)
        U1 = U + k1
        U2 = 0.75 * U + 0.25 * (U1 + rhs(U1))
        return (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + rhs(U2))
    if rk == "RK4":
        k2 = rhs(U + 0.5 * k1)
        k3 = rhs(U + 0.5 * k2)
        k4 = rhs(U + k3)
        return U + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    # RK5 — Dormand-Prince (5th order, 6 stages)
    k2 = rhs(U + (1.0/5.0) * k1)
    k3 = rhs(U + (3.0/40.0) * k1 + (9.0/40.0) * k2)
    k4 = rhs(U + (44.0/45.0) * k1 - (56.0/15.0) * k2 + (32.0/9.0) * k3)
    k5 = rhs(U + (19372.0/6561.0) * k1 - (25360.0/2187.0) * k2
             + (64448.0/6561.0) * k3 - (212.0/729.0) * k4)
    k6 = rhs(U + (9017.0/3168.0) * k1 - (355.0/33.0) * k2
             + (46732.0/5247.0) * k3 + (49.0/176.0) * k4
             - (5103.0/18656.0) * k5)
    return U + (35.0/384.0) * k1 + (500.0/1113.0) * k3 \
             + (125.0/192.0) * k4 - (2187.0/6784.0) * k5 \
             + (11.0/84.0) * k6


def _apply_positivity_floor(U: np.ndarray) -> tuple[int, float]:
    """Clip density and internal energy in place; return (cells, energy added).

    The correction is not conservative: it adds mass and energy.  Callers
    must report it rather than hide it.
    """
    rho_low = U[0] < _FLOOR
    U[0] = np.maximum(U[0], _FLOOR)
    ke = 0.5 * U[1]**2 / U[0]
    e_low = U[2] - ke < _FLOOR
    E_old = U[2].copy()
    U[2] = np.where(e_low, ke + _FLOOR, U[2])
    return int(np.count_nonzero(rho_low | e_low)), float((U[2] - E_old).sum())


def run_simulation(
    config: SimulationConfig,
    scheme: NumericalScheme,
    n_snapshots: int = 1,
    progress_callback: Callable[[float], None] | None = None,
    time_integrator: str | None = None,
    positivity_floor: bool = False,
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
       spatial order to an RK method (order 1 → RK1, 2 → RK2, 3 and above
       → SSP-RK3; JST uses RK4).

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
    positivity_floor : bool, optional
        If True, clip density and internal energy to a small floor after each
        step so that schemes without positivity guarantee keep running.  The
        correction is not conservative; it is counted in ``floor_events`` and
        ``floor_energy``.  Off by default so that each scheme shows its own
        behaviour, failure included.

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

    if mesh.n_cells < scheme.n_ghost:
        raise ValueError(
            f"{scheme.name} needs {scheme.n_ghost} ghost cells per side and at "
            f"least as many interior cells; got n_cells={mesh.n_cells}"
        )

    dx = mesh.dx
    x, U = initial_state(config)

    t = 0.0
    t_final = tc.t_final

    rk = select_time_integrator(scheme, time_integrator)

    # Snapshot times
    if n_snapshots > 0:
        snapshot_times = set(
            t_final * (i + 1) / n_snapshots for i in range(n_snapshots)
        )
    else:
        snapshot_times = set()
    snapshot_times.add(t_final)

    result = SolverResult(x=x, scheme_name=scheme.name, config=config)
    cfl = min(tc.cfl, scheme.max_cfl)
    result.cfl = cfl

    t_start = time.perf_counter()

    def diverge(reason: str) -> None:
        result.status = "diverged"
        warnings.warn(
            f"Solution diverged at t={t:.6e} ({reason}). "
            f"Returning the last physical state.",
            RuntimeWarning,
            stacklevel=3,
        )

    cfl_warned = False
    while t < t_final:
        # Compute time step
        with np.errstate(invalid="ignore"):
            s_max = max_wave_speed(U, gas.gamma)
        if not np.isfinite(s_max):
            diverge("non-finite wave speed")
            break
        if s_max < 1e-14:
            s_max = 1e-14
        if tc.dt_fixed is not None:
            dt = tc.dt_fixed
            cfl_eff = dt * s_max / dx
            if cfl_eff > scheme.max_cfl and not cfl_warned:
                cfl_warned = True
                warnings.warn(
                    f"dt_fixed={dt:.3e} gives CFL={cfl_eff:.2f} at t={t:.3e}, above "
                    f"the limit {scheme.max_cfl} of {scheme.name}: stability is "
                    f"not guaranteed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            dt = cfl * dx / s_max

        # Don't overshoot.  A flux whose dissipation scales with dx/dt
        # (Lax-Friedrichs) keeps the nominal step: with the shortened one, a
        # tiny final step would still apply a full averaging.  Schemes that
        # embed the time integration (Lax-Wendroff) need the actual step.
        dt_flux = dt
        if t + dt > t_final:
            dt = t_final - t
            if scheme.time_integral_included:
                dt_flux = dt

        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            U_new = _rk_step(
                U, rk,
                lambda v: _compute_rhs(v, scheme, gas, dx, dt, config.bc_type, dt_flux),
            )
        if not np.isfinite(U_new).all():
            diverge("NaN/Inf in the updated state")
            break

        if positivity_floor:
            n_cells, energy = _apply_positivity_floor(U_new)
            result.floor_events += n_cells
            result.floor_energy += energy * dx
        rho_new, _, p_new = conservative_to_primitive(U_new, gas.gamma)
        if (rho_new <= 0.0).any() or (p_new <= 0.0).any():
            diverge("negative density or pressure")
            break
        U = U_new

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

    # On divergence, or if no snapshot was recorded, end with the last physical state
    if result.status == "diverged" or not result.snapshots:
        result.times.append(t)
        result.snapshots.append(U.copy())

    if result.floor_events:
        warnings.warn(
            f"Positivity floor applied {result.floor_events} times "
            f"(energy injected: {result.floor_energy:.3e}). The scheme alone "
            f"produced negative density or pressure.",
            RuntimeWarning,
            stacklevel=2,
        )

    result.wall_time = time.perf_counter() - t_start
    return result
