"""HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver.

HLLC extends HLL by adding a **third wave** (the contact wave S*) to the
two-wave HLL model.  The solution structure now consists of four regions
separated by three waves (S_L, S*, S_R), exactly mirroring the wave structure
of the exact Riemann problem.

The contact wave speed S* is determined from momentum conservation across the
outer waves (Rankine-Hugoniot conditions).  The two intermediate star states
share the same pressure and velocity but have different densities, correctly
capturing contact discontinuities that HLL smears.

HLLC offers the best quality-to-cost ratio among approximate Riemann solvers:
it resolves all three wave families without requiring the Newton iteration of
the exact Godunov solver or the entropy fix needed by Roe.  It is the
recommended flux for combination with higher-order reconstructions (MUSCL-HLLC,
WENO3-HLLC, WENO5-HLLC).

The current implementation uses a Python loop over interfaces due to the
four-case conditional structure.

References:
    [Toro, 2009] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid
        Dynamics, 3rd ed., Springer, Section 10.4.
    [Batten et al., 1997] P. Batten, N. Clarke, C. Lambert, D.M. Causon,
        SIAM J. Sci. Comput., 18(6), pp. 1553-1570.

See also: docs/04_flux_schemes.md, Section 4.4.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import compute_flux, conservative_to_primitive, sound_speed
from euler1d.schemes.base import NumericalScheme


class HLLC(NumericalScheme):
    """HLLC (HLL-Contact) three-wave approximate Riemann solver.

    Extends HLL with a contact wave S* that separates two intermediate star
    states sharing the same pressure and velocity but different densities.
    This correctly captures contact discontinuities that HLL smears.

    The recommended solver for general use and for combination with
    higher-order reconstructions (MUSCL-HLLC, WENO3-HLLC, WENO5-HLLC).

    See also: docs/04_flux_schemes.md, Section 4.4.
    """

    @property
    def name(self) -> str:
        return "HLLC"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute the HLLC flux at all interfaces.

        Outer wave speeds S_L and S_R use Davis estimates.  The contact speed
        S* is derived from momentum conservation across the outer waves.
        Intermediate star states are built from the Rankine-Hugoniot jump
        conditions.  A four-case selection (left, star-left, star-right,
        right) determines the flux at each interface.

        Interfaces are processed in a Python loop due to the conditional
        branching on four regions.

        See also: docs/04_flux_schemes.md, Section 4.4.
        """
        FL = compute_flux(UL, gas.gamma)
        FR = compute_flux(UR, gas.gamma)

        rhoL, uL, pL = conservative_to_primitive(UL, gas.gamma)
        rhoR, uR, pR = conservative_to_primitive(UR, gas.gamma)
        aL = sound_speed(rhoL, pL, gas.gamma)
        aR = sound_speed(rhoR, pR, gas.gamma)

        # Davis estimates
        SL = np.minimum(uL - aL, uR - aR)
        SR = np.maximum(uL + aL, uR + aR)

        # Contact wave speed S*
        S_star = (
            (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR))
            / (rhoL * (SL - uL) - rhoR * (SR - uR))
        )

        n_intf = FL.shape[1]
        F = np.empty((3, n_intf))

        for k in range(n_intf):
            if SL[k] >= 0.0:
                F[:, k] = FL[:, k]
            elif SR[k] <= 0.0:
                F[:, k] = FR[:, k]
            elif S_star[k] >= 0.0:
                # F*_L
                coeff = rhoL[k] * (SL[k] - uL[k]) / (SL[k] - S_star[k])
                U_starL = coeff * np.array([
                    1.0,
                    S_star[k],
                    UL[2, k] / rhoL[k]
                    + (S_star[k] - uL[k])
                    * (S_star[k] + pL[k] / (rhoL[k] * (SL[k] - uL[k]))),
                ])
                F[:, k] = FL[:, k] + SL[k] * (U_starL - UL[:, k])
            else:
                # F*_R
                coeff = rhoR[k] * (SR[k] - uR[k]) / (SR[k] - S_star[k])
                U_starR = coeff * np.array([
                    1.0,
                    S_star[k],
                    UR[2, k] / rhoR[k]
                    + (S_star[k] - uR[k])
                    * (S_star[k] + pR[k] / (rhoR[k] * (SR[k] - uR[k]))),
                ])
                F[:, k] = FR[:, k] + SR[k] * (U_starR - UR[:, k])

        return F
