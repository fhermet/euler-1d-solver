"""HLL (Harten-Lax-van Leer) approximate Riemann solver.

The HLL scheme improves upon Rusanov by modelling the Riemann problem with
**two waves** (S_L and S_R) instead of one, yielding a single intermediate
state between them.  Wave speeds are estimated using the Davis bounds.  The
three-case flux formula selects the left flux, the right flux, or the
intermediate HLL flux depending on the sign of S_L and S_R.

With only one intermediate state, HLL cannot represent the contact
discontinuity: the density profile is averaged across the contact, resulting
in a smeared contact region.  HLLC (Section 4.4) corrects this by adding a
third wave for the contact.

The implementation is fully vectorised using numpy boolean masks, making it
one of the fastest solvers.  It is composable with higher-order
reconstructions.

References:
    [Harten, Lax, van Leer, 1983] A. Harten, P.D. Lax, B. van Leer,
        SIAM Review, 25(1), pp. 35-61.
    [Davis, 1988] S.F. Davis, SIAM J. Sci. Stat. Comput., 9(3), pp. 445-473.

See also: docs/04_flux_schemes.md, Section 4.3.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import compute_flux, conservative_to_primitive, sound_speed
from euler1d.schemes.base import NumericalScheme


def davis_speeds(UL: np.ndarray, UR: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Davis estimates of the slowest and fastest wave speeds (S_L, S_R).

    S_L = min(u_L - a_L, u_R - a_R), S_R = max(u_L + a_L, u_R + a_R).
    Shared by HLL and HLLC, and drawn by the finite-volume walkthrough.
    """
    rhoL, uL, pL = conservative_to_primitive(UL, gamma)
    rhoR, uR, pR = conservative_to_primitive(UR, gamma)
    aL = sound_speed(rhoL, pL, gamma)
    aR = sound_speed(rhoR, pR, gamma)
    return np.minimum(uL - aL, uR - aR), np.maximum(uL + aL, uR + aR)


class HLL(NumericalScheme):
    """HLL (Harten-Lax-van Leer) two-wave approximate Riemann solver.

    Models the Riemann fan with two bounding waves (S_L, S_R) and a single
    intermediate state.  Wave speeds are estimated with the Davis bounds.
    Less dissipative than Rusanov but smears contact discontinuities because
    the contact wave is absent from the model.

    Fully vectorised using numpy boolean masks.

    See also: docs/04_flux_schemes.md, Section 4.3.
    """

    @property
    def name(self) -> str:
        return "HLL"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute the HLL flux at all interfaces.

        Three cases are evaluated via boolean masks: supersonic flow from the
        left (S_L >= 0), supersonic from the right (S_R <= 0), or subsonic
        (intermediate HLL flux).  Wave speeds use the Davis estimates.

        See also: docs/04_flux_schemes.md, Section 4.3.
        """
        FL = compute_flux(UL, gas.gamma)
        FR = compute_flux(UR, gas.gamma)

        SL, SR = davis_speeds(UL, UR, gas.gamma)

        F = np.empty_like(FL)

        # Vectorized HLL flux
        left_mask = SL >= 0.0
        right_mask = SR <= 0.0
        mid_mask = ~left_mask & ~right_mask

        F[:, left_mask] = FL[:, left_mask]
        F[:, right_mask] = FR[:, right_mask]

        if np.any(mid_mask):
            sl = SL[mid_mask]
            sr = SR[mid_mask]
            denom = sr - sl
            F[:, mid_mask] = (
                sr * FL[:, mid_mask] - sl * FR[:, mid_mask]
                + sl * sr * (UR[:, mid_mask] - UL[:, mid_mask])
            ) / denom

        return F
