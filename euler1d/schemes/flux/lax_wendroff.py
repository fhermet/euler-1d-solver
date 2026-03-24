"""Lax-Wendroff scheme (Richtmyer two-step predictor-corrector).

The Lax-Wendroff scheme achieves second-order accuracy in both space and time
using a two-stage approach.  The predictor stage estimates the state at the
interface at the half time step using a Lax-Friedrichs-type average.  The
corrector stage evaluates the physical flux at this predicted state.

Because the predictor already embeds a half time step, the temporal integration
is included in the flux itself.  The property ``time_integral_included`` is set
to True, which forces the time integrator to use a single stage (RK1).  Using
a multi-stage Runge-Kutta method on top would advance the solution incorrectly.

As a second-order centred scheme without limiting, Lax-Wendroff produces
dispersive oscillations (Gibbs phenomenon) near discontinuities.  It is not
composable with higher-order reconstructions.

References:
    [Lax, Wendroff, 1960] P.D. Lax, B. Wendroff, Comm. Pure Appl. Math.,
        13, pp. 217-237.
    [Richtmyer, 1963] R.D. Richtmyer, NCAR Technical Note 63-2.

See also: docs/04_flux_schemes.md, Section 4.7.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import compute_flux
from euler1d.schemes.base import NumericalScheme


class LaxWendroff(NumericalScheme):
    """Lax-Wendroff (Richtmyer two-step) centred scheme.

    A predictor-corrector scheme that is second order in both space and time.
    The predictor estimates the half-time-step state at the interface; the
    corrector evaluates the physical flux at that predicted state.

    Because the temporal integration is embedded in the predictor step,
    ``time_integral_included`` is True, forcing the solver to use RK1.
    Not composable with higher-order reconstructions.

    See also: docs/04_flux_schemes.md, Section 4.7.
    """

    @property
    def name(self) -> str:
        return "Lax-Wendroff"

    @property
    def order(self) -> int:
        return 2

    @property
    def scheme_type(self) -> str:
        return "centré"

    @property
    def time_integral_included(self) -> bool:
        return True

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute the Lax-Wendroff flux at all interfaces.

        The Richtmyer predictor averages the left and right states and
        advances by half a time step using the flux difference.  The
        corrector returns the physical flux evaluated at this predicted
        state.  The result already includes temporal integration, so the
        time-stepping loop must use a single Euler stage (RK1).

        See also: docs/04_flux_schemes.md, Section 4.7.
        """
        FL = compute_flux(UL, gas.gamma)
        FR = compute_flux(UR, gas.gamma)
        # Richtmyer predictor: half-step state at interface
        U_half = 0.5 * (UL + UR) - 0.5 * (dt / dx) * (FR - FL)
        # Corrector: flux from predicted state
        return compute_flux(U_half, gas.gamma)
