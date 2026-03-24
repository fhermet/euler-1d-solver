"""Lax-Friedrichs (global) centred scheme.

The Lax-Friedrichs scheme is historically one of the earliest conservative
schemes for hyperbolic conservation laws.  It is a centred scheme: the
numerical flux is the average of the left and right physical fluxes minus a
dissipation term proportional to the global ratio dx/dt.

Unlike Rusanov, which uses the local maximum wave speed, Lax-Friedrichs uses
the global quantity dx/dt as the dissipation coefficient.  Since the CFL
condition ensures dx/dt >= max(|u|+a), this makes Lax-Friedrichs the most
dissipative stable scheme.  Discontinuities are heavily smeared.

The scheme is not composable with higher-order reconstructions because it is
not formulated as a Riemann solver.  Its dissipation also depends on the time
step, coupling the spatial and temporal discretisations.

References:
    [Lax, 1954] P.D. Lax, Comm. Pure Appl. Math., 7, pp. 159-193.

See also: docs/04_flux_schemes.md, Section 4.6.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import compute_flux
from euler1d.schemes.base import NumericalScheme


class LaxFriedrichs(NumericalScheme):
    """Lax-Friedrichs (global) centred scheme.

    Centred flux average with dissipation proportional to the global ratio
    dx/dt rather than a local wave speed.  This makes it the most dissipative
    stable scheme.  Not composable with higher-order reconstructions.

    See also: docs/04_flux_schemes.md, Section 4.6.
    """

    @property
    def name(self) -> str:
        return "Lax-Friedrichs"

    @property
    def scheme_type(self) -> str:
        return "centré"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute the Lax-Friedrichs flux at all interfaces.

        The dissipation coefficient is dx/dt (global, not based on local wave
        speeds), which always exceeds the maximum characteristic speed under
        the CFL condition.  This results in maximal numerical diffusion.

        See also: docs/04_flux_schemes.md, Section 4.6.
        """
        FL = compute_flux(UL, gas.gamma)
        FR = compute_flux(UR, gas.gamma)
        return 0.5 * (FL + FR) - 0.5 * (dx / dt) * (UR - UL)
