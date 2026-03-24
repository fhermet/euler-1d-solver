"""Roe linearised Riemann solver with Harten-Hyman entropy fix.

The Roe scheme linearises the Euler equations around specially chosen average
states (Roe averages, weighted by sqrt(rho)) so that the resulting constant-
coefficient linear system satisfies a discrete Rankine-Hugoniot condition.
The linearised Riemann problem is then solved exactly, yielding three waves
with their own speeds and amplitudes.

The numerical flux takes the form of a centred average minus a dissipation
term where each wave is damped at its own characteristic speed.  This
selective dissipation makes Roe significantly sharper than Rusanov or HLL for
contact discontinuities and shocks.

Without modification, the Roe scheme can admit non-physical expansion shocks
when an eigenvalue passes through zero.  The Harten-Hyman entropy fix
regularises the absolute eigenvalues in transonic rarefaction regions,
restoring the correct entropy-satisfying behaviour.

The implementation is fully vectorised (no Python loop).  It is composable
with higher-order reconstructions.

References:
    [Roe, 1981] P.L. Roe, J. Comput. Phys., 43(2), pp. 357-372.
    [Harten, Hyman, 1983] A. Harten, J.M. Hyman, J. Comput. Phys., 50,
        pp. 235-269.

See also: docs/04_flux_schemes.md, Section 4.5.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import compute_flux, conservative_to_primitive, sound_speed
from euler1d.schemes.base import NumericalScheme


def _roe_common(UL, UR, gas):
    """Compute Roe averages, eigenvalues, wave strengths and physical fluxes.

    Returns a dict with all quantities needed by both Roe variants.
    """
    gamma = gas.gamma
    gm1 = gas.gm1

    FL = compute_flux(UL, gamma)
    FR = compute_flux(UR, gamma)

    rhoL, uL, pL = conservative_to_primitive(UL, gamma)
    rhoR, uR, pR = conservative_to_primitive(UR, gamma)
    aL = sound_speed(rhoL, pL, gamma)
    aR = sound_speed(rhoR, pR, gamma)
    HL = (UL[2] + pL) / rhoL
    HR = (UR[2] + pR) / rhoR

    sqrtL = np.sqrt(rhoL)
    sqrtR = np.sqrt(rhoR)
    denom = sqrtL + sqrtR

    u_roe = (sqrtL * uL + sqrtR * uR) / denom
    H_roe = (sqrtL * HL + sqrtR * HR) / denom
    a_roe_sq = gm1 * (H_roe - 0.5 * u_roe**2)
    a_roe = np.sqrt(np.maximum(a_roe_sq, 1e-30))

    lam1 = u_roe - a_roe
    lam2 = u_roe
    lam3 = u_roe + a_roe

    drho = rhoR - rhoL
    du = uR - uL
    dp = pR - pL

    rho_roe = sqrtL * sqrtR
    alpha1 = (dp - rho_roe * a_roe * du) / (2.0 * a_roe**2)
    alpha2 = drho - dp / (a_roe**2)
    alpha3 = (dp + rho_roe * a_roe * du) / (2.0 * a_roe**2)

    return {
        "FL": FL, "FR": FR,
        "uL": uL, "aL": aL, "uR": uR, "aR": aR,
        "u_roe": u_roe, "H_roe": H_roe, "a_roe": a_roe,
        "lam1": lam1, "lam2": lam2, "lam3": lam3,
        "alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3,
    }


def _roe_flux(FL, FR, u_roe, H_roe, a_roe, lam1, lam2, lam3, alpha1, alpha2, alpha3):
    """Assemble the Roe flux from corrected absolute eigenvalues."""
    n_intf = FL.shape[1]
    dissipation = np.zeros((3, n_intf))

    dissipation[0] = lam1 * alpha1 + lam2 * alpha2 + lam3 * alpha3
    dissipation[1] = (
        lam1 * alpha1 * (u_roe - a_roe)
        + lam2 * alpha2 * u_roe
        + lam3 * alpha3 * (u_roe + a_roe)
    )
    dissipation[2] = (
        lam1 * alpha1 * (H_roe - u_roe * a_roe)
        + lam2 * alpha2 * 0.5 * u_roe**2
        + lam3 * alpha3 * (H_roe + u_roe * a_roe)
    )

    return 0.5 * (FL + FR) - 0.5 * dissipation


class RoeNoFix(NumericalScheme):
    """Roe linearised Riemann solver WITHOUT entropy fix.

    Pure Roe scheme (1981) using sqrt(rho)-weighted averages.  The
    conservative-variable jump is decomposed into three waves, each damped
    at its own characteristic speed.

    WARNING: without entropy correction, this scheme can produce non-physical
    expansion shocks (entropy-violating solutions) in transonic rarefaction
    regions.  Use the ``double_rarefaction`` test case to observe this defect.

    See also: docs/04_flux_schemes.md, Section 4.5.
    """

    @property
    def name(self) -> str:
        return "Roe (sans correction)"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        d = _roe_common(UL, UR, gas)
        return _roe_flux(
            d["FL"], d["FR"],
            d["u_roe"], d["H_roe"], d["a_roe"],
            np.abs(d["lam1"]), np.abs(d["lam2"]), np.abs(d["lam3"]),
            d["alpha1"], d["alpha2"], d["alpha3"],
        )


class Roe(NumericalScheme):
    """Roe linearised Riemann solver with Harten-Hyman entropy fix.

    Linearises the Euler equations around sqrt(rho)-weighted averages (Roe
    averages) that satisfy a discrete Rankine-Hugoniot condition.  The
    conservative-variable jump is decomposed into three waves, each damped
    at its own characteristic speed, yielding selective dissipation that is
    much sharper than Rusanov or HLL.

    The Harten-Hyman entropy fix prevents non-physical expansion shocks by
    regularising eigenvalues that change sign across the interface.

    Fully vectorised, no Python loop.

    See also: docs/04_flux_schemes.md, Section 4.5.
    """

    @property
    def name(self) -> str:
        return "Roe"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        d = _roe_common(UL, UR, gas)

        lam1, lam2, lam3 = d["lam1"], d["lam2"], d["lam3"]
        uL, aL, uR, aR = d["uL"], d["aL"], d["uR"], d["aR"]

        # Harten-Hyman entropy fix
        eps1 = np.maximum(0.0, np.maximum(lam1 - (uL - aL), (uR - aR) - lam1))
        eps3 = np.maximum(0.0, np.maximum(lam3 - (uL + aL), (uR + aR) - lam3))
        lam1 = np.where(np.abs(lam1) < eps1, eps1, np.abs(lam1))
        lam2 = np.abs(lam2)
        lam3 = np.where(np.abs(lam3) < eps3, eps3, np.abs(lam3))

        return _roe_flux(
            d["FL"], d["FR"],
            d["u_roe"], d["H_roe"], d["a_roe"],
            lam1, lam2, lam3,
            d["alpha1"], d["alpha2"], d["alpha3"],
        )
