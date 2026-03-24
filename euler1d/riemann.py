"""Exact Riemann solver for the 1D Euler equations (perfect gas).

Solves the Riemann problem: given two constant states (left and right)
separated by a discontinuity, computes the exact time-dependent solution
composed of three waves (two acoustic, one contact). The solution procedure
is:

1. Solve a nonlinear scalar equation for the star-region pressure p*
   using Newton-Raphson iteration (PVRS initial guess).
2. Derive the star-region velocity u* and densities rho*_L, rho*_R.
3. Sample the full solution at any position using the self-similar
   variable xi = x/t and an optimized decision tree.

The sampling is vectorized using numpy boolean masks instead of a
Python loop, enabling efficient evaluation on large meshes.

This module is the computational backbone of the Godunov scheme
(see euler1d/schemes/flux/godunov.py) and provides reference solutions
for all Riemann-type test cases.

Theory:
    See docs/02_riemann_problem.md for the complete mathematical derivation
    of every formula implemented here, including the pressure equation,
    Rankine-Hugoniot relations, rarefaction fan profiles, and sampling
    decision tree.

Reference:
    E. F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
    3rd Edition, Springer, 2009. Chapters 4-5.
"""

from __future__ import annotations

import math

import numpy as np


def _sound_speed_scalar(rho: float, p: float, gamma: float) -> float:
    return math.sqrt(gamma * p / rho)


def _f_shock(p: float, rhoK: float, pK: float, gamma: float) -> float:
    """Shock branch of the pressure function for one side of the Riemann problem.

    Computes the velocity change across a shock connecting state K to the
    star region. Active when p* > p_K (compression).

    See docs/02_riemann_problem.md §2.2 and §2.5 for the Rankine-Hugoniot
    derivation and the role of coefficients A_K and B_K.
    """
    AK = 2.0 / ((gamma + 1.0) * rhoK)
    BK = (gamma - 1.0) / (gamma + 1.0) * pK
    return (p - pK) * math.sqrt(AK / (p + BK))


def _f_rare(p: float, rhoK: float, pK: float, gamma: float) -> float:
    """Rarefaction branch of the pressure function for one side.

    Computes the velocity change across an isentropic rarefaction fan
    connecting state K to the star region. Active when p* <= p_K (expansion).

    See docs/02_riemann_problem.md §2.3 and §2.5 for the derivation from
    Riemann invariants and the isentropic relation.
    """
    aK = _sound_speed_scalar(rhoK, pK, gamma)
    return (2.0 * aK / (gamma - 1.0)) * ((p / pK) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _f_side(p: float, rhoK: float, pK: float, gamma: float) -> float:
    """Select the shock or rarefaction branch based on pressure ratio.

    Dispatches to _f_shock (compression, p > p_K) or _f_rare (expansion,
    p <= p_K). The two branches join continuously at p = p_K.

    See docs/02_riemann_problem.md §2.5 for the unified pressure equation.
    """
    if p > pK:
        return _f_shock(p, rhoK, pK, gamma)
    return _f_rare(p, rhoK, pK, gamma)


def _f_total(
    p: float,
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
) -> float:
    """Total pressure function whose root gives p*.

    Evaluates f(p) = f_L(p) + f_R(p) + (u_R - u_L). The root f(p*) = 0
    determines the star-region pressure where left and right wave
    contributions are compatible with the velocity jump.

    See docs/02_riemann_problem.md §2.5 for the derivation.
    """
    return _f_side(p, rhol, pl, gamma) + _f_side(p, rhor, pr, gamma) + (ur - ul)


def _df_dp_side(p: float, rhoK: float, pK: float, gamma: float) -> float:
    """Derivative of the pressure function for one side (shock or rarefaction).

    Used by the Newton iteration to compute f'(p). Each branch has a
    closed-form analytical derivative ensuring quadratic convergence.

    See docs/02_riemann_problem.md §2.6 for the derivative formulas.
    """
    if p > pK:
        AK = 2.0 / ((gamma + 1.0) * rhoK)
        BK = (gamma - 1.0) / (gamma + 1.0) * pK
        sqrt_term = math.sqrt(AK / (p + BK))
        return sqrt_term * (1.0 - 0.5 * (p - pK) / (p + BK))
    aK = _sound_speed_scalar(rhoK, pK, gamma)
    return (1.0 / (rhoK * aK)) * (p / pK) ** (-(gamma + 1.0) / (2.0 * gamma))


def _df_total(
    p: float,
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
) -> float:
    """Total derivative of the pressure function for Newton iteration.

    Sums the left and right side derivatives. The velocity difference
    (u_R - u_L) is a constant and does not contribute to the derivative.

    See docs/02_riemann_problem.md §2.6.
    """
    return _df_dp_side(p, rhol, pl, gamma) + _df_dp_side(p, rhor, pr, gamma)


def _pressure_guess_pvrs(
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
) -> float:
    """Primitive Variable Riemann Solver (PVRS) initial guess for p*.

    Provides a linearized estimate of the star-region pressure by
    averaging the left and right pressures with an acoustic correction
    based on the velocity jump and impedance-weighted average. This
    gives a good starting point for Newton iteration.

    See docs/02_riemann_problem.md §2.6 for the physical interpretation
    of each term.
    """
    aL = _sound_speed_scalar(rhol, pl, gamma)
    aR = _sound_speed_scalar(rhor, pr, gamma)
    pPV = 0.5 * (pl + pr) - 0.5 * (ur - ul) * (rhol * aL + rhor * aR) / (rhol + rhor)
    return max(1e-14, pPV)


def _vacuum_will_form(
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
) -> bool:
    """Check if the velocity difference exceeds the critical threshold for vacuum.

    When the left and right states separate faster than the maximum
    expansion speed that rarefaction waves can sustain, no positive
    star-region pressure exists and a vacuum forms.

    See docs/02_riemann_problem.md §2.6 for the vacuum criterion and its
    physical meaning.
    """
    aL = _sound_speed_scalar(rhol, pl, gamma)
    aR = _sound_speed_scalar(rhor, pr, gamma)
    return (ur - ul) >= (2.0 / (gamma - 1.0)) * (aL + aR)


def solve_p_star(
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Solve for the star-region pressure p* using Newton-Raphson iteration.

    Starting from a PVRS initial guess, iteratively finds the root of the
    nonlinear pressure equation f(p*) = f_L(p*) + f_R(p*) + (u_R - u_L) = 0.
    This equation encodes the compatibility between the left and right waves:
    the velocity changes across both waves must bridge the initial velocity
    difference. Convergence is quadratic, typically requiring 3-6 iterations.

    Raises ValueError if a vacuum would form (states separating too fast
    for any positive pressure solution to exist).

    See docs/02_riemann_problem.md §2.5-§2.6 for the pressure equation
    derivation, Newton update formulas, and convergence criterion.

    Reference: Toro (2009), Chapter 4, §4.3.

    Parameters
    ----------
    rhol, ul, pl : float
        Left state primitive variables (density, velocity, pressure).
    rhor, ur, pr : float
        Right state primitive variables.
    gamma : float
        Ratio of specific heats.
    tol : float
        Relative convergence tolerance for Newton iteration.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    float
        Star-region pressure p*.
    """
    if _vacuum_will_form(rhol, ul, pl, rhor, ur, pr, gamma):
        raise ValueError(
            "Vacuum forms in this Riemann problem: no positive p_star exists."
        )

    p = _pressure_guess_pvrs(rhol, ul, pl, rhor, ur, pr, gamma)
    for _ in range(max_iter):
        f = _f_total(p, rhol, ul, pl, rhor, ur, pr, gamma)
        df = _df_total(p, rhol, ul, pl, rhor, ur, pr, gamma)
        if abs(df) < 1e-20:
            break
        p_new = p - f / df
        p_new = max(1e-14, p_new)
        if abs(p_new - p) < tol * max(1.0, abs(p_new)):
            return p_new
        p = p_new
    return p


def compute_u_star(
    pstar: float,
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
) -> float:
    """Compute the star-region velocity u* from the known p*.

    The velocity in the star region is the same on both sides of the
    contact discontinuity. It is obtained by averaging the left and right
    velocity contributions through their respective waves (shock or
    rarefaction). Physically, u* is the speed at which the contact
    surface separating the two intermediate states propagates.

    See docs/02_riemann_problem.md §2.5 for the derivation.

    Reference: Toro (2009), Chapter 4, §4.2.

    Parameters
    ----------
    pstar : float
        Star-region pressure (from solve_p_star).
    rhol, ul, pl : float
        Left state primitive variables.
    rhor, ur, pr : float
        Right state primitive variables.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Star-region velocity u*.
    """
    fL = _f_side(pstar, rhol, pl, gamma)
    fR = _f_side(pstar, rhor, pr, gamma)
    return 0.5 * (ul + ur) + 0.5 * (fR - fL)


def star_region_densities(
    pstar: float,
    rhol: float, pl: float,
    rhor: float, pr: float,
    gamma: float,
) -> tuple[float, float]:
    """Compute star-region densities on both sides of the contact discontinuity.

    The density in the star region differs on each side because the contact
    discontinuity allows a density jump while pressure and velocity remain
    continuous. The formula depends on the wave type:
    - Across a shock: Rankine-Hugoniot jump relation (compression ratio
      bounded by the gas-specific maximum).
    - Across a rarefaction: isentropic (Poisson) relation.

    See docs/02_riemann_problem.md §2.7 for both formulas and their
    physical interpretation.

    Reference: Toro (2009), Chapter 4, §4.4.

    Parameters
    ----------
    pstar : float
        Star-region pressure.
    rhol, pl : float
        Left state density and pressure.
    rhor, pr : float
        Right state density and pressure.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    tuple[float, float]
        (rhoL_star, rhoR_star) densities in the left and right star regions.
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    if pstar > pl:
        r = pstar / pl
        rhoL_star = rhol * (r + gm1 / gp1) / (gm1 / gp1 * r + 1.0)
    else:
        rhoL_star = rhol * (pstar / pl) ** (1.0 / gamma)

    if pstar > pr:
        r = pstar / pr
        rhoR_star = rhor * (r + gm1 / gp1) / (gm1 / gp1 * r + 1.0)
    else:
        rhoR_star = rhor * (pstar / pr) ** (1.0 / gamma)

    return rhoL_star, rhoR_star


def solve_riemann(
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
) -> tuple[float, float, float, float]:
    """Solve the exact Riemann problem for two given left and right states.

    This is the main entry point that orchestrates the full solution:
    1. Newton iteration for p* (solve_p_star),
    2. Velocity u* from p* (compute_u_star),
    3. Densities on both sides of the contact (star_region_densities).

    The four returned quantities fully characterize the star region and
    determine the wave structure (shock or rarefaction on each side).

    See docs/02_riemann_problem.md for the complete theory, and §2.9 for
    the mapping between concepts and functions.

    Reference: Toro (2009), Chapter 4.

    Parameters
    ----------
    rhol, ul, pl : float
        Left state primitive variables.
    rhor, ur, pr : float
        Right state primitive variables.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    tuple[float, float, float, float]
        (pstar, ustar, rhoL_star, rhoR_star).
    """
    pstar = solve_p_star(rhol, ul, pl, rhor, ur, pr, gamma)
    ustar = compute_u_star(pstar, rhol, ul, pl, rhor, ur, pr, gamma)
    rhoL_star, rhoR_star = star_region_densities(pstar, rhol, pl, rhor, pr, gamma)
    return pstar, ustar, rhoL_star, rhoR_star


def sample_riemann(
    x: np.ndarray,
    t: float,
    rhol: float, ul: float, pl: float,
    rhor: float, ur: float, pr: float,
    gamma: float,
    x0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the exact Riemann solution at an array of positions (vectorized).

    Given the left/right states and a time t > 0, evaluates the exact
    self-similar solution at every position in x. The algorithm:
    1. Solves for the star region (p*, u*, rho*_L, rho*_R).
    2. For each position, computes the similarity variable xi = (x - x0)/t.
    3. Traverses the sampling decision tree (left/right of contact,
       shock/rarefaction, inside/outside fan) using boolean masks for
       vectorized assignment.

    At t = 0, returns the piecewise constant initial condition directly.

    See docs/02_riemann_problem.md §2.7 for the complete sampling decision
    tree and the rarefaction fan profiles.

    Reference: Toro (2009), Chapter 4, §4.5.

    Parameters
    ----------
    x : array_like
        Positions at which to evaluate the solution.
    t : float
        Time at which to evaluate (must be >= 0).
    rhol, ul, pl : float
        Left state primitive variables.
    rhor, ur, pr : float
        Right state primitive variables.
    gamma : float
        Ratio of specific heats.
    x0 : float
        Initial discontinuity position (default 0.0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (rho, u, p) arrays of the same shape as x.
    """
    x = np.asarray(x, dtype=float)

    if t == 0.0:
        rho = np.where(x < x0, rhol, rhor)
        u = np.where(x < x0, ul, ur)
        p = np.where(x < x0, pl, pr)
        return rho, u, p

    pstar, ustar, rhoL_star, rhoR_star = solve_riemann(
        rhol, ul, pl, rhor, ur, pr, gamma
    )

    aL = _sound_speed_scalar(rhol, pl, gamma)
    aR = _sound_speed_scalar(rhor, pr, gamma)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    xi = (x - x0) / t

    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    # --- left side (xi < ustar) ---
    left = xi < ustar

    if pstar > pl:
        # Left shock
        sL = ul - aL * math.sqrt(gp1 / (2.0 * gamma) * (pstar / pl - 1.0) + 1.0)
        undisturbed_L = left & (xi < sL)
        star_L = left & (xi >= sL)
        rho[undisturbed_L] = rhol
        u[undisturbed_L] = ul
        p[undisturbed_L] = pl
        rho[star_L] = rhoL_star
        u[star_L] = ustar
        p[star_L] = pstar
    else:
        # Left rarefaction
        headL = ul - aL
        aL_star = aL * (pstar / pl) ** (gm1 / (2.0 * gamma))
        tailL = ustar - aL_star

        undisturbed_L = left & (xi < headL)
        fan_L = left & (xi >= headL) & (xi <= tailL)
        star_L = left & (xi > tailL)

        rho[undisturbed_L] = rhol
        u[undisturbed_L] = ul
        p[undisturbed_L] = pl

        rho[star_L] = rhoL_star
        u[star_L] = ustar
        p[star_L] = pstar

        if np.any(fan_L):
            xi_fan = xi[fan_L]
            u_fan = (2.0 / gp1) * (aL + 0.5 * gm1 * ul + xi_fan)
            a_fan = aL - 0.5 * gm1 * (u_fan - ul)
            rho[fan_L] = rhol * (a_fan / aL) ** (2.0 / gm1)
            u[fan_L] = u_fan
            p[fan_L] = pl * (a_fan / aL) ** (2.0 * gamma / gm1)

    # --- right side (xi >= ustar) ---
    right = ~left

    if pstar > pr:
        # Right shock
        sR = ur + aR * math.sqrt(gp1 / (2.0 * gamma) * (pstar / pr - 1.0) + 1.0)
        undisturbed_R = right & (xi > sR)
        star_R = right & (xi <= sR)
        rho[undisturbed_R] = rhor
        u[undisturbed_R] = ur
        p[undisturbed_R] = pr
        rho[star_R] = rhoR_star
        u[star_R] = ustar
        p[star_R] = pstar
    else:
        # Right rarefaction
        headR = ur + aR
        aR_star = aR * (pstar / pr) ** (gm1 / (2.0 * gamma))
        tailR = ustar + aR_star

        undisturbed_R = right & (xi > headR)
        fan_R = right & (xi >= tailR) & (xi <= headR)
        star_R = right & (xi < tailR)

        rho[undisturbed_R] = rhor
        u[undisturbed_R] = ur
        p[undisturbed_R] = pr

        rho[star_R] = rhoR_star
        u[star_R] = ustar
        p[star_R] = pstar

        if np.any(fan_R):
            xi_fan = xi[fan_R]
            u_fan = (2.0 / gp1) * (-aR + 0.5 * gm1 * ur + xi_fan)
            a_fan = aR + 0.5 * gm1 * (u_fan - ur)
            rho[fan_R] = rhor * (a_fan / aR) ** (2.0 / gm1)
            u[fan_R] = u_fan
            p[fan_R] = pr * (a_fan / aR) ** (2.0 * gamma / gm1)

    return rho, u, p


def sample_at_interface(
    rhoL: float, uL: float, pL: float,
    rhoR: float, uR: float, pR: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Sample the Riemann solution at the interface xi = x/t = 0 (Godunov flux).

    Evaluates the exact solution at the cell interface, which is the
    value needed by the Godunov scheme to compute the numerical flux.
    By the self-similar property, the solution at xi = 0 depends only on
    the left and right states and not on the mesh spacing or time step.

    This is a convenience wrapper around sample_riemann that evaluates
    at a single point x = 0, t = 1.

    See docs/02_riemann_problem.md §2.7 for the sampling procedure and
    §2.8 for the role in the Godunov scheme.

    Reference: Toro (2009), Chapter 4, §4.5; Chapter 6 (Godunov method).

    Parameters
    ----------
    rhoL, uL, pL : float
        Left state primitive variables.
    rhoR, uR, pR : float
        Right state primitive variables.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    tuple[float, float, float]
        (rho, u, p) at the interface.
    """
    x = np.array([0.0])
    rho, u, p = sample_riemann(x, 1.0, rhoL, uL, pL, rhoR, uR, pR, gamma, x0=0.0)
    return float(rho[0]), float(u[0]), float(p[0])
