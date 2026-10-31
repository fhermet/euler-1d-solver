"""Godunov scheme using the exact Riemann solver.

The Godunov scheme is the historical reference for finite-volume methods on
hyperbolic systems.  At every cell interface it solves the **exact Riemann
problem** (Chapter 2) between the left and right states, then evaluates the
physical flux at the self-similar solution sampled at the interface (x/t = 0).

Because the exact solver involves a Newton iteration for the intermediate
pressure, the scheme is more expensive than approximate solvers such as HLLC
or Roe.  Newton runs on all interfaces at once (``riemann.sample_interfaces``).
When the interface states generate a vacuum, the exact solution with a vacuum
region is used (Toro 2009, §4.6).

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
from euler1d.riemann import sample_interfaces
from euler1d.schemes.base import NumericalScheme


class Godunov(NumericalScheme):
    """Godunov flux using the exact Riemann solver.

    At each interface the exact Riemann problem is solved via Newton
    iteration on the intermediate pressure, and the physical flux is
    evaluated at the self-similar solution sampled at x/t = 0.

    This is the most accurate first-order flux but also the most expensive
    because of the iterative solver.

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

        All interfaces are processed at once (vectorised Newton iteration).

        See also: docs/04_flux_schemes.md, Section 4.1.
        """
        rhoL, uL, pL = conservative_to_primitive(UL, gas.gamma)
        rhoR, uR, pR = conservative_to_primitive(UR, gas.gamma)
        rho, u, p = sample_interfaces(rhoL, uL, pL, rhoR, uR, pR, gas.gamma)
        E = p / (gas.gamma - 1.0) + 0.5 * rho * u**2
        return np.array([rho * u, rho * u**2 + p, u * (E + p)])
