"""Rusanov (local Lax-Friedrichs) approximate Riemann solver.

The Rusanov scheme is the simplest upwind flux.  It models the Riemann problem
with a **single wave** travelling at the local maximum speed
S_max = max(|u_L|+a_L, |u_R|+a_R).  The numerical flux is the average of the
left and right physical fluxes minus a dissipation term proportional to the
jump in conservative variables scaled by S_max.

Using one speed for all three wave families makes the scheme very robust but
also very dissipative: the contact discontinuity, which propagates at speed u,
is damped at a rate proportional to |u| + a instead of |u|.  Contact
discontinuities are therefore heavily smeared compared to HLLC or Roe.

The implementation is fully vectorised (no Python loop over interfaces), which
makes Rusanov one of the cheapest solvers per interface.  It is composable with
higher-order reconstructions (MUSCL, ENO, WENO).

References:
    [Rusanov, 1961] V.V. Rusanov, Zh. Vychisl. Mat. i Mat. Fiz., 1(2),
        pp. 267-279.
    [Toro, 2009] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid
        Dynamics, 3rd ed., Springer, Section 10.5.

See also: docs/04_flux_schemes.md, Section 4.2.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import compute_flux, conservative_to_primitive, sound_speed
from euler1d.schemes.base import NumericalScheme


class Rusanov(NumericalScheme):
    """Rusanov (local Lax-Friedrichs) approximate Riemann solver.

    The simplest upwind flux: a centred average of the physical fluxes minus
    a dissipation term scaled by the local maximum wave speed.  Very robust
    but very dissipative, especially for contact discontinuities.

    Fully vectorised, no Python loop.

    See also: docs/04_flux_schemes.md, Section 4.2.
    """

    @property
    def name(self) -> str:
        return "Rusanov"

    @property
    def scheme_type(self) -> str:
        return "centré"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute the Rusanov flux at all interfaces.

        The flux is the centred average of the left and right physical fluxes
        minus a dissipation term proportional to the conservative-variable jump
        scaled by the maximum of |u|+a on each side.  All operations are
        vectorised over interfaces.

        See also: docs/04_flux_schemes.md, Section 4.2.
        """
        FL = compute_flux(UL, gas.gamma)
        FR = compute_flux(UR, gas.gamma)

        rhoL, uL, pL = conservative_to_primitive(UL, gas.gamma)
        rhoR, uR, pR = conservative_to_primitive(UR, gas.gamma)
        aL = sound_speed(rhoL, pL, gas.gamma)
        aR = sound_speed(rhoR, pR, gas.gamma)

        S_max = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
        return 0.5 * (FL + FR) - 0.5 * S_max[np.newaxis, :] * (UR - UL)
