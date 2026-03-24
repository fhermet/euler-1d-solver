"""Predefined test cases for 1D Euler equations.

Each function returns a fully configured :class:`SimulationConfig` ready to
pass to :func:`euler1d.solver.run_simulation`.  Two categories of test cases
are provided:

* **Riemann problems** (discontinuous initial data): Sod shock tube, Lax test,
  double rarefaction, stationary contact, near vacuum, two shocks (Toro
  test 4).  The exact solution is computed by the exact Riemann solver
  (:mod:`euler1d.riemann`).
* **Shock/wave interaction**: Shu-Osher (Mach 3 shock in sinusoidal density).
  Reference solution computed at high resolution (WENO5-Z, 2000 cells).
* **Smooth problems** (analytic exact solution): entropy wave, acoustic wave.
  These supply an ``exact_fn(x, t, gamma)`` used for convergence-order
  verification.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import (
    GasProperties,
    MeshConfig,
    PrimitiveState,
    RiemannProblem,
    SimulationConfig,
    SmoothProblem,
    TimeConfig,
)


def sod_shock_tube(n_cells: int = 200) -> SimulationConfig:
    """Classic Sod shock tube problem (Section 8.1).

    Models the burst of a membrane separating a high-pressure chamber
    (left: rho=1, u=0, p=1) from a low-pressure chamber (right: rho=0.125,
    u=0, p=0.1) with the discontinuity at x=0.5.

    Expected wave structure (left to right):
        1. Left-going rarefaction fan
        2. Contact discontinuity
        3. Right-going shock wave

    This is the most commonly used benchmark for compressible flow solvers.
    It tests basic wave capturing, contact resolution, and shock sharpness.

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).

    Returns
    -------
    SimulationConfig
        Complete configuration with transmissive BC, t_final=0.2, gamma=1.4.

    References
    ----------
    G.A. Sod, *J. Comput. Phys.*, 27(1), 1978.
    """
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=0.2, cfl=0.9),
        problem=RiemannProblem(
            left=PrimitiveState(rho=1.0, u=0.0, p=1.0),
            right=PrimitiveState(rho=0.125, u=0.0, p=0.1),
            x_discontinuity=0.5,
        ),
        bc_type="transmissive",
    )


def lax_test(n_cells: int = 200) -> SimulationConfig:
    """Lax test problem -- more severe than Sod (Section 8.2).

    Initial states: (rho, u, p)_L = (0.445, 0.698, 3.528),
    (rho, u, p)_R = (0.5, 0, 0.571).  The non-zero left velocity creates
    stronger wave interactions than the Sod problem.

    This test validates robustness with asymmetric states and stronger shocks.

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).

    Returns
    -------
    SimulationConfig
        Complete configuration with transmissive BC, t_final=0.14, gamma=1.4.
    """
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=0.14, cfl=0.9),
        problem=RiemannProblem(
            left=PrimitiveState(rho=0.445, u=0.698, p=3.528),
            right=PrimitiveState(rho=0.5, u=0.0, p=0.571),
            x_discontinuity=0.5,
        ),
        bc_type="transmissive",
    )


def double_rarefaction(n_cells: int = 200) -> SimulationConfig:
    """123 problem -- double rarefaction stress test (Section 8.3).

    Initial states: (rho, u, p)_L = (1, -2, 0.4),
    (rho, u, p)_R = (1, 2, 0.4).  Opposing velocities pull the gas apart,
    generating two symmetric rarefaction fans and a very low pressure region
    at the centre of the domain.

    This test validates:
        * Pressure positivity preservation under near-vacuum conditions.
        * Robustness of the scheme (some methods produce negative pressure).

    If (u_R - u_L) >= 2/(gamma-1)*(a_L + a_R), a true vacuum forms and the
    exact Riemann solver raises ``ValueError`` (see Ch. 2, Section 2.6).

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).

    Returns
    -------
    SimulationConfig
        Complete configuration with transmissive BC, t_final=0.15, gamma=1.4.
    """
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=0.15, cfl=0.9),
        problem=RiemannProblem(
            left=PrimitiveState(rho=1.0, u=-2.0, p=0.4),
            right=PrimitiveState(rho=1.0, u=2.0, p=0.4),
            x_discontinuity=0.5,
        ),
        bc_type="transmissive",
    )


def stationary_contact(n_cells: int = 200) -> SimulationConfig:
    """Stationary contact discontinuity -- pure diffusion test (Section 8.5).

    Initial states: (rho, u, p)_L = (1, 0, 1), (rho, u, p)_R = (0.125, 0, 1).
    Velocity and pressure are identical on both sides; only density has a jump
    at x = 0.5.

    The exact solution is stationary for all time: the contact discontinuity
    does not move.  Any spreading of the density profile is purely due to
    numerical diffusion introduced by the scheme.

    This is the cleanest test for comparing numerical diffusion across flux
    solvers, since there are no shocks or rarefaction fans to interfere.

    Expected differences:
        * Rusanov: maximum diffusion (dissipation ~ |u| + a, but a > 0).
        * HLL: strong diffusion (no contact wave in the model).
        * HLLC/Roe: minimal diffusion (contact wave resolved, dissipation ~ |u| = 0).
        * Godunov: exact (no diffusion on a stationary contact).

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).

    Returns
    -------
    SimulationConfig
        Complete configuration with transmissive BC, t_final=0.5, gamma=1.4.
    """
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=0.5, cfl=0.9),
        problem=RiemannProblem(
            left=PrimitiveState(rho=1.0, u=0.0, p=1.0),
            right=PrimitiveState(rho=0.125, u=0.0, p=1.0),
            x_discontinuity=0.5,
        ),
        bc_type="transmissive",
    )


def shu_osher(n_cells: int = 200) -> SimulationConfig:
    """Shu-Osher problem -- shock/entropy wave interaction (Section 8.7).

    A Mach 3 shock propagates rightward into a medium with sinusoidal density
    perturbations.  Behind the shock, the interaction generates complex
    small-scale oscillations that are physical (not numerical artifacts).

    Initial conditions (domain [-5, 5]):
        Left of x = -4:  (rho, u, p) = (3.857143, 2.629369, 10.33333)
        Right of x = -4: rho = 1 + 0.2*sin(5*x), u = 0, p = 1

    This is NOT a Riemann problem: the right state has spatially varying
    density.  The exact solution is computed as a high-resolution reference
    (WENO5-Z + HLLC on 2000 cells).

    This is the best test case for demonstrating why high-order schemes
    matter: MUSCL smears the post-shock oscillations, while WENO5 captures
    them faithfully.

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).

    Returns
    -------
    SimulationConfig
        Complete configuration with transmissive BC, t_final=1.8, gamma=1.4.

    References
    ----------
    C.-W. Shu, S. Osher, *J. Comput. Phys.*, 83(1), 1989, pp. 32-78.
    """
    gamma = 1.4

    # Post-shock state (Mach 3 shock in gamma=1.4 gas)
    rho_L, u_L, p_L = 3.857143, 2.629369, 10.33333
    x_shock = -4.0

    def init_fn(x, gamma):
        rho = np.where(x < x_shock, rho_L, 1.0 + 0.2 * np.sin(5.0 * x))
        u = np.where(x < x_shock, u_L, 0.0)
        p = np.where(x < x_shock, p_L, 1.0)
        return rho, u, p

    # Reference solution cache (computed once at high resolution)
    _ref_cache = {}

    def exact_fn(x, t, gamma):
        if "rho" not in _ref_cache:
            from euler1d.solver import run_simulation
            from euler1d.schemes import get_scheme
            from euler1d.physics import conservative_to_primitive

            ref_config = SimulationConfig(
                gas=GasProperties(gamma=gamma),
                mesh=MeshConfig(x_min=-5.0, x_max=5.0, n_cells=2000),
                time=TimeConfig(t_final=t, cfl=0.9),
                problem=SmoothProblem(init_fn=init_fn, exact_fn=lambda x, t, g: (x, x, x), label="shu_osher"),
                bc_type="transmissive",
            )
            ref_result = run_simulation(ref_config, get_scheme("wenoz5-hllc"))
            ref_rho, ref_u, ref_p = conservative_to_primitive(
                ref_result.snapshots[-1], gamma
            )
            _ref_cache["x"] = ref_result.x
            _ref_cache["rho"] = ref_rho
            _ref_cache["u"] = ref_u
            _ref_cache["p"] = ref_p

        # Interpolate reference onto requested grid
        rho = np.interp(x, _ref_cache["x"], _ref_cache["rho"])
        u = np.interp(x, _ref_cache["x"], _ref_cache["u"])
        p = np.interp(x, _ref_cache["x"], _ref_cache["p"])
        return rho, u, p

    return SimulationConfig(
        gas=GasProperties(gamma=gamma),
        mesh=MeshConfig(x_min=-5.0, x_max=5.0, n_cells=n_cells),
        time=TimeConfig(t_final=1.8, cfl=0.9),
        problem=SmoothProblem(init_fn=init_fn, exact_fn=exact_fn, label="shu_osher"),
        bc_type="transmissive",
    )


def near_vacuum(n_cells: int = 200) -> SimulationConfig:
    """Near-vacuum test -- extreme double rarefaction close to vacuum limit (Section 8.5).

    Initial states: (rho, u, p)_L = (1, -3.5, 0.4),
    (rho, u, p)_R = (1, 3.5, 0.4).  The opposing velocities are at 93% of the
    vacuum formation threshold (2/(gamma-1)*(a_L + a_R) = 7.483 for these
    states), creating an extremely low-pressure region at the centre with
    p* ~ 10^{-13}.

    This is a more severe version of ``double_rarefaction`` (which uses u = ±2,
    reaching only 53% of the vacuum threshold).

    This test validates:
        * Pressure and density positivity preservation under near-vacuum.
        * Robustness of flux schemes with extreme low-density regions.
        * Correctness of rarefaction fan computation near the vacuum limit.

    Many schemes require positivity floors to survive this test case.

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).

    Returns
    -------
    SimulationConfig
        Complete configuration with transmissive BC, t_final=0.1, gamma=1.4.
    """
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=0.1, cfl=0.9),
        problem=RiemannProblem(
            left=PrimitiveState(rho=1.0, u=-3.5, p=0.4),
            right=PrimitiveState(rho=1.0, u=3.5, p=0.4),
            x_discontinuity=0.5,
        ),
        bc_type="transmissive",
    )


def two_shocks(n_cells: int = 200) -> SimulationConfig:
    """Toro test 4 -- collision of two strong shocks (Section 8.4).

    Initial states: (rho, u, p)_L = (5.99924, 19.5975, 460.894),
    (rho, u, p)_R = (5.99242, -6.19633, 46.0950).  Both sides have large
    inward velocities, creating two strong shocks that collide near the centre,
    producing a very high-pressure, high-density intermediate state.

    Expected wave structure (left to right):
        1. Left-going shock
        2. Contact discontinuity
        3. Right-going shock

    This test validates:
        * Correct capture of strong shock interactions.
        * Resolution of the contact discontinuity between the two shocks.
        * Robustness under extreme post-shock conditions (p* ~ 1700).

    This is the symmetric counterpart of ``double_rarefaction``: opposing
    velocities compress (instead of expand) the gas.

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).

    Returns
    -------
    SimulationConfig
        Complete configuration with transmissive BC, t_final=0.035, gamma=1.4.

    References
    ----------
    E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
    3rd ed., Springer, 2009, Test 4, p. 334.
    """
    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=0.035, cfl=0.9),
        problem=RiemannProblem(
            left=PrimitiveState(rho=5.99924, u=19.5975, p=460.894),
            right=PrimitiveState(rho=5.99242, u=-6.19633, p=46.0950),
            x_discontinuity=0.4,
        ),
        bc_type="transmissive",
    )


def entropy_wave(n_cells: int = 200, epsilon: float = 0.2) -> SimulationConfig:
    """Entropy wave: sinusoidal density perturbation advected at constant velocity (Section 8.4).

    Initial condition: rho = 1 + epsilon*sin(2*pi*x), u = u0 = 1, p = p0 = 1.
    The exact solution is pure advection of the density profile at speed u0;
    velocity and pressure remain constant (pure entropy mode, Ch. 7, Section 7.2).

    With periodic BC and t_final = 1.0, the wave completes one full domain
    traversal, so the final solution equals the initial condition.  This makes
    it the primary test case for convergence-order verification: an order-N
    scheme should produce error proportional to (dx)^N.

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).
    epsilon : float, optional
        Amplitude of the density perturbation (default 0.2).

    Returns
    -------
    SimulationConfig
        Complete configuration with periodic BC, t_final=1.0, gamma=1.4.
        The ``SmoothProblem`` includes ``exact_fn`` for error computation.
    """
    u0 = 1.0
    p0 = 1.0

    def init_fn(x, gamma):
        rho = 1.0 + epsilon * np.sin(2.0 * np.pi * x)
        u = np.full_like(x, u0)
        p = np.full_like(x, p0)
        return rho, u, p

    def exact_fn(x, t, gamma):
        rho = 1.0 + epsilon * np.sin(2.0 * np.pi * (x - u0 * t))
        u = np.full_like(x, u0)
        p = np.full_like(x, p0)
        return rho, u, p

    return SimulationConfig(
        gas=GasProperties(gamma=1.4),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=1.0, cfl=0.9),
        problem=SmoothProblem(init_fn=init_fn, exact_fn=exact_fn, label="entropy_wave"),
        bc_type="periodic",
    )


def acoustic_wave(n_cells: int = 200, epsilon: float = 1e-4) -> SimulationConfig:
    """Right-going acoustic wave: small isentropic perturbation (Section 8.5).

    Initial perturbation aligned with the right acoustic eigenvector (Ch. 1,
    Section 1.4) around base state (rho0, u0, p0) = (1, 0, 1):
        delta_rho = epsilon * sin(2*pi*x)
        delta_u   = (a0/rho0) * epsilon * sin(2*pi*x)
        delta_p   = a0**2 * epsilon * sin(2*pi*x)

    The linearised solution propagates at speed a0 = sqrt(gamma*p0/rho0)
    without deformation.  With periodic BC and t_final = 1/a0, the wave
    completes one domain traversal.

    Unlike the entropy wave, all three fields (rho, u, p) vary simultaneously,
    testing convergence on coupled acoustic modes.  Epsilon must be small
    (default 1e-4) for the linearised solution to remain valid.

    Parameters
    ----------
    n_cells : int, optional
        Number of mesh cells (default 200).
    epsilon : float, optional
        Perturbation amplitude (default 1e-4).

    Returns
    -------
    SimulationConfig
        Complete configuration with periodic BC, t_final=1/a0, gamma=1.4.
        The ``SmoothProblem`` includes ``exact_fn`` for error computation.
    """
    gamma = 1.4
    rho0, u0, p0 = 1.0, 0.0, 1.0
    a0 = np.sqrt(gamma * p0 / rho0)

    def init_fn(x, gamma):
        rho = rho0 + epsilon * np.sin(2.0 * np.pi * x)
        u = (a0 / rho0) * epsilon * np.sin(2.0 * np.pi * x)
        p = p0 + a0**2 * epsilon * np.sin(2.0 * np.pi * x)
        return rho, u, p

    def exact_fn(x, t, gamma):
        rho = rho0 + epsilon * np.sin(2.0 * np.pi * (x - a0 * t))
        u = (a0 / rho0) * epsilon * np.sin(2.0 * np.pi * (x - a0 * t))
        p = p0 + a0**2 * epsilon * np.sin(2.0 * np.pi * (x - a0 * t))
        return rho, u, p

    return SimulationConfig(
        gas=GasProperties(gamma=gamma),
        mesh=MeshConfig(x_min=0.0, x_max=1.0, n_cells=n_cells),
        time=TimeConfig(t_final=1.0 / a0, cfl=0.9),
        problem=SmoothProblem(init_fn=init_fn, exact_fn=exact_fn, label="acoustic_wave"),
        bc_type="periodic",
    )
