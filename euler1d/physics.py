"""Equation of state, primitive/conservative conversions, and physical flux.

This module implements the fundamental thermodynamic and kinematic relations
for a perfect gas governed by the 1D Euler equations. It provides conversions
between conservative variables U = (rho, rho*u, E) and primitive variables
W = (rho, u, p), the physical flux vector F(U), the sound speed, and the
maximum wave speed used for CFL time-step control.

All array inputs and outputs follow the convention of shape (3, N) for
state vectors and (N,) for scalar fields, where N is the number of cells.

See Also
--------
docs/01_euler_equations.md : Chapter 1, sections 1.1 through 1.4 for the
    governing equations, equation of state, variable conversions, and wave
    speeds.

References
----------
[Toro, 2009] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid
    Dynamics, 3rd ed., Springer, section 1.3.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def primitive_to_conservative(
    rho: np.ndarray, u: np.ndarray, p: np.ndarray, gamma: float
) -> np.ndarray:
    """Convert primitive variables to conservative variables U(3, N).

    Transforms the physically intuitive representation (density, velocity,
    pressure) into the conserved quantities (density, momentum, total energy)
    that appear in the Euler conservation laws. The total energy includes both
    the internal (thermal) energy derived from pressure via the equation of
    state and the kinetic energy of the bulk fluid motion.

    Parameters
    ----------
    rho : ndarray, shape (N,)
        Density — mass per unit volume of the gas.
    u : ndarray, shape (N,)
        Velocity — bulk fluid velocity.
    p : ndarray, shape (N,)
        Pressure — thermodynamic pressure from molecular thermal agitation.
    gamma : float
        Ratio of specific heats characterizing the gas.

    Returns
    -------
    U : ndarray, shape (3, N)
        Conservative state vector: row 0 is density, row 1 is momentum
        (density times velocity), row 2 is total energy (internal plus
        kinetic energy per unit volume).

    See Also
    --------
    docs/01_euler_equations.md : section 1.3 — variable conversion formulas.

    References
    ----------
    [Toro, 2009] section 1.3.
    """
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.array([rho, rho * u, E])


def conservative_to_primitive(
    U: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert conservative U(3, N) to primitive (rho, u, p).

    Recovers the physically meaningful quantities (density, velocity, pressure)
    from the conserved variables. Velocity is obtained by dividing momentum by
    density. Pressure is recovered from the equation of state by subtracting
    kinetic energy from total energy.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative state vector: density, momentum, total energy.
    gamma : float
        Ratio of specific heats characterizing the gas.

    Returns
    -------
    rho : ndarray, shape (N,)
        Density — mass per unit volume.
    u : ndarray, shape (N,)
        Velocity — bulk fluid velocity.
    p : ndarray, shape (N,)
        Pressure — thermodynamic pressure.

    See Also
    --------
    docs/01_euler_equations.md : section 1.3 — variable conversion formulas.

    References
    ----------
    [Toro, 2009] section 1.3.
    """
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u**2)
    return rho, u, p


def compute_flux(U: np.ndarray, gamma: float) -> np.ndarray:
    """Compute physical flux F(3, N) from conservative variables U(3, N).

    Evaluates the Euler flux vector F(U) which represents the rate of transport
    of each conserved quantity through a cross-section. The mass flux is the
    momentum itself. The momentum flux combines convective transport and
    pressure forces. The energy flux accounts for convective energy transport
    plus the work done by pressure forces.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative state vector: density, momentum, total energy.
    gamma : float
        Ratio of specific heats characterizing the gas.

    Returns
    -------
    F : ndarray, shape (3, N)
        Physical flux vector: mass flux, momentum flux (convection plus
        pressure), energy flux (total enthalpy times velocity).

    See Also
    --------
    docs/01_euler_equations.md : section 1.1 — governing equations and flux
        interpretation.

    References
    ----------
    [Toro, 2009] section 1.3.
    """
    rho, u, p = conservative_to_primitive(U, gamma)
    E = U[2]
    return np.array([rho * u, rho * u**2 + p, u * (E + p)])


def sound_speed(rho: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the local speed of sound in a perfect gas.

    The sound speed is the propagation velocity of small isentropic pressure
    perturbations. It determines the acoustic wave speeds (eigenvalues of the
    Euler system) and thus controls the CFL stability condition and the
    numerical domain of dependence.

    Parameters
    ----------
    rho : ndarray, shape (N,)
        Density — mass per unit volume.
    p : ndarray, shape (N,)
        Pressure — thermodynamic pressure.
    gamma : float
        Ratio of specific heats characterizing the gas.

    Returns
    -------
    a : ndarray, shape (N,)
        Local sound speed.

    See Also
    --------
    docs/01_euler_equations.md : section 1.2 — equation of state and sound
        speed derivation.

    References
    ----------
    [Toro, 2009] section 1.3.
    """
    return np.sqrt(gamma * p / rho)


def max_wave_speed(U: np.ndarray, gamma: float) -> float:
    """Compute maximum wave speed across the domain.

    Returns the largest absolute eigenvalue of the Euler system over all cells,
    which is max(|u| + a). This is the fastest speed at which information can
    propagate in the solution and is used to determine the adaptive time step
    via the CFL condition: dt = CFL * dx / max_wave_speed.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative state vector: density, momentum, total energy.
    gamma : float
        Ratio of specific heats characterizing the gas.

    Returns
    -------
    s_max : float
        Maximum wave speed over the entire domain.

    See Also
    --------
    docs/01_euler_equations.md : section 1.4 — eigenvalues of the Euler system.

    References
    ----------
    [Toro, 2009] section 1.3.
    """
    rho, u, p = conservative_to_primitive(U, gamma)
    a = sound_speed(rho, p, gamma)
    return float(np.max(np.abs(u) + a))


def cell_average(
    prim_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    x: np.ndarray,
    dx: float,
    gamma: float,
    n_points: int = 5,
) -> np.ndarray:
    """Cell averages U(3, N) of a pointwise primitive field, by Gauss quadrature.

    A finite volume scheme evolves cell averages.  Initialising, or comparing
    against, point values at cell centres instead introduces an O(dx^2)
    error, which hides the order of any scheme above 2.  The quadrature uses
    *n_points* Gauss-Legendre points per cell (exact for polynomials of
    degree 2*n_points - 1).

    Parameters
    ----------
    prim_fn : callable
        x -> (rho, u, p) evaluated pointwise.
    x : ndarray, shape (N,)
        Cell centres.
    dx : float
        Cell width.
    gamma : float
        Ratio of specific heats.
    n_points : int, optional
        Number of Gauss points per cell (default 5).

    Returns
    -------
    U_bar : ndarray, shape (3, N)
        Cell averages of the conservative variables.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    U_bar = np.zeros((3, len(x)))
    for xi, w in zip(nodes, weights):
        rho, u, p = prim_fn(x + 0.5 * dx * xi)
        U_bar += 0.5 * w * primitive_to_conservative(rho, u, p, gamma)
    return U_bar
