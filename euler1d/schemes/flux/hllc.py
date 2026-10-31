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

The four-case selection is vectorised with boolean masks: all star states
are computed for every interface, then the relevant one is picked.

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
from euler1d.physics import compute_flux, conservative_to_primitive
from euler1d.schemes.flux.hll import davis_speeds
from euler1d.schemes.base import NumericalScheme


def contact_speed(
    UL: np.ndarray, UR: np.ndarray, gamma: float, SL: np.ndarray, SR: np.ndarray
) -> np.ndarray:
    """HLLC contact wave speed S*, from momentum conservation across S_L and S_R."""
    rhoL, uL, pL = conservative_to_primitive(UL, gamma)
    rhoR, uR, pR = conservative_to_primitive(UR, gamma)
    return (
        (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR))
        / (rhoL * (SL - uL) - rhoR * (SR - uR))
    )


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

        All four candidate fluxes are evaluated on every interface and the
        right one is selected by mask (no Python loop).

        See also: docs/04_flux_schemes.md, Section 4.4.
        """
        FL = compute_flux(UL, gas.gamma)
        FR = compute_flux(UR, gas.gamma)

        rhoL, uL, pL = conservative_to_primitive(UL, gas.gamma)
        rhoR, uR, pR = conservative_to_primitive(UR, gas.gamma)

        SL, SR = davis_speeds(UL, UR, gas.gamma)
        S_star = contact_speed(UL, UR, gas.gamma, SL, SR)

        # Star fluxes F*_K = F_K + S_K (U*_K - U_K), K = L, R.  Where a branch
        # is not selected its value may be meaningless (division by zero when
        # S_K = S*), hence the errstate; np.where discards it.
        with np.errstate(divide="ignore", invalid="ignore"):
            F_starL = _star_flux(UL, FL, rhoL, uL, pL, SL, S_star)
            F_starR = _star_flux(UR, FR, rhoR, uR, pR, SR, S_star)

        return np.where(
            SL >= 0.0, FL,
            np.where(SR <= 0.0, FR, np.where(S_star >= 0.0, F_starL, F_starR)),
        )


def _star_flux(U, F, rho, u, p, S, S_star):
    """HLLC star-region flux on one side (Toro 2009, §10.4.2)."""
    coeff = rho * (S - u) / (S - S_star)
    U_star = coeff * np.array([
        np.ones_like(S_star),
        S_star,
        U[2] / rho + (S_star - u) * (S_star + p / (rho * (S - u))),
    ])
    return F + S * (U_star - U)
