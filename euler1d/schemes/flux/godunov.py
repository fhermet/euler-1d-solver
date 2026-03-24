"""Godunov scheme using the exact Riemann solver.

The Godunov scheme is the historical reference for finite-volume methods on
hyperbolic systems.  At every cell interface it solves the **exact Riemann
problem** (Chapter 2) between the left and right states, then evaluates the
physical flux at the self-similar solution sampled at the interface (x/t = 0).

Because the exact solver involves a Newton iteration for the intermediate
pressure, the scheme is significantly more expensive than approximate solvers
such as HLLC or Roe.  In the current implementation, interfaces are processed
sequentially in a Python loop (the exact sampler is not vectorised), which
further increases the wall-clock cost.

Despite using the exact Riemann solution, the overall spatial accuracy remains
first order because the reconstruction is piecewise constant.  Combining
Godunov with a higher-order reconstruction (MUSCL, WENO) is possible but
rarely done in practice; HLLC or Roe are preferred for that purpose.

References:
    [Godunov, 1959] S.K. Godunov, Mat. Sb., 47(3), pp. 271-306.
    [Toro, 2009] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid
        Dynamics, 3rd ed., Springer, Section 6.2.

See also: docs/04_flux_schemes.md, Section 4.1.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import conservative_to_primitive
from euler1d.riemann import sample_at_interface
from euler1d.schemes.base import NumericalScheme


class Godunov(NumericalScheme):
    """Godunov flux using the exact Riemann solver.

    At each interface the exact Riemann problem is solved via Newton
    iteration on the intermediate pressure, and the physical flux is
    evaluated at the self-similar solution sampled at x/t = 0.

    This is the most accurate first-order flux but also the most expensive
    due to the iterative solver and the per-interface Python loop.

    See also: docs/04_flux_schemes.md, Section 4.1.
    """

    @property
    def name(self) -> str:
        return "Godunov"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute the Godunov flux by solving the exact Riemann problem.

        For each interface, the exact Riemann solver determines the
        intermediate pressure via Newton iteration, identifies the wave
        types (shock or rarefaction), and samples the solution at the
        interface (x/t = 0) to evaluate the physical flux.

        The interfaces are processed sequentially because the exact
        sampler is not vectorised.

        See also: docs/04_flux_schemes.md, Section 4.1.
        """
        rhoL, uL, pL = conservative_to_primitive(UL, gas.gamma)
        rhoR, uR, pR = conservative_to_primitive(UR, gas.gamma)

        n_intf = UL.shape[1]
        F = np.empty((3, n_intf))

        for k in range(n_intf):
            rho_s, u_s, p_s = sample_at_interface(
                rhoL[k], uL[k], pL[k],
                rhoR[k], uR[k], pR[k],
                gas.gamma,
            )
            E_s = p_s / (gas.gamma - 1.0) + 0.5 * rho_s * u_s**2
            F[0, k] = rho_s * u_s
            F[1, k] = rho_s * u_s**2 + p_s
            F[2, k] = u_s * (E_s + p_s)

        return F
