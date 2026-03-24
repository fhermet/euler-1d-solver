"""AUSM+ and AUSM+-up flux schemes (Advection Upstream Splitting Method).

Both schemes split the numerical flux into two physically distinct parts:

    F_{1/2} = (mass flux) * Phi_{1/2} + P_{1/2}

where the mass flux dot{m}_{1/2} transports convective quantities (rho, rho*u,
rho*H) and P_{1/2} is the pressure flux (0, p, 0)^T.  Each part uses its own
splitting based on the local interface Mach number.

AUSM+ (Liou, 1996) is the base scheme.  AUSM+-up (Liou, 2006) adds three
corrections to improve robustness near discontinuities and at low Mach:
    - Mach dissipation (M_p): pressure-difference term in the interface Mach.
    - Pressure dissipation (p_u): velocity-difference term in the interface
      pressure.
    - Velocity scaling (f_a): adapts the dissipation level to the local Mach
      regime, reducing it in supersonic flow where it is not needed.

Both implementations are fully vectorised (no Python loop over interfaces).

References:
    [Liou, 1996] M.-S. Liou, "A sequel to AUSM: AUSM+",
        J. Comput. Phys., 129, pp. 364-382.
    [Liou, 2006] M.-S. Liou, "A sequel to AUSM, Part II: AUSM+-up for
        all speeds", J. Comput. Phys., 214, pp. 137-170.

See also: docs/04_flux_schemes.md, Section 4.9.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import conservative_to_primitive, sound_speed
from euler1d.schemes.base import NumericalScheme


def _mach_split_plus(M: np.ndarray) -> np.ndarray:
    """M+ splitting: smooth polynomial for |M| <= 1, upwind for |M| > 1."""
    return np.where(
        np.abs(M) >= 1.0,
        0.5 * (M + np.abs(M)),                          # (M + |M|) / 2
        0.25 * (M + 1.0)**2 + 0.125 * (M**2 - 1.0)**2,  # AUSM+ polynomial
    )


def _mach_split_minus(M: np.ndarray) -> np.ndarray:
    """M- splitting: smooth polynomial for |M| <= 1, upwind for |M| > 1."""
    return np.where(
        np.abs(M) >= 1.0,
        0.5 * (M - np.abs(M)),                            # (M - |M|) / 2
        -0.25 * (M - 1.0)**2 - 0.125 * (M**2 - 1.0)**2,  # AUSM+ polynomial
    )


def _pressure_split_plus(M: np.ndarray) -> np.ndarray:
    """P+ splitting for the pressure flux."""
    return np.where(
        np.abs(M) >= 1.0,
        0.5 * (1.0 + np.sign(M)),
        0.25 * (M + 1.0)**2 * (2.0 - M) + 0.1875 * M * (M**2 - 1.0)**2,
    )


def _pressure_split_minus(M: np.ndarray) -> np.ndarray:
    """P- splitting for the pressure flux."""
    return np.where(
        np.abs(M) >= 1.0,
        0.5 * (1.0 - np.sign(M)),
        0.25 * (M - 1.0)**2 * (2.0 + M) - 0.1875 * M * (M**2 - 1.0)**2,
    )


class AUSMPlus(NumericalScheme):
    """AUSM+ flux scheme (Liou, 1996).

    Splits the interface flux into a convective part (mass flux times
    transported quantities) and a pressure part, each using smooth
    polynomial splittings of the local Mach number.

    The interface speed of sound is taken as the arithmetic average
    a_{1/2} = (a_L + a_R) / 2, and the interface Mach number is
    M_{1/2} = M+_L + M-_R where M_L = u_L / a_{1/2}, M_R = u_R / a_{1/2}.

    Fully vectorised, composable with higher-order reconstructions.

    See also: docs/04_flux_schemes.md, Section 4.9.
    """

    @property
    def name(self) -> str:
        return "AUSM+"

    @property
    def scheme_type(self) -> str:
        return "FVS"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        gamma = gas.gamma

        rhoL, uL, pL = conservative_to_primitive(UL, gamma)
        rhoR, uR, pR = conservative_to_primitive(UR, gamma)
        aL = sound_speed(rhoL, pL, gamma)
        aR = sound_speed(rhoR, pR, gamma)
        HL = (UL[2] + pL) / rhoL  # total enthalpy
        HR = (UR[2] + pR) / rhoR

        # Interface speed of sound
        a_half = 0.5 * (aL + aR)

        # Left and right Mach numbers
        ML = uL / a_half
        MR = uR / a_half

        # Interface Mach number
        M_half = _mach_split_plus(ML) + _mach_split_minus(MR)

        # Mass flux
        m_dot = a_half * np.where(
            M_half >= 0.0,
            M_half * rhoL,
            M_half * rhoR,
        )

        # Convective flux: m_dot * Phi (upwind selection of transported quantities)
        n_intf = UL.shape[1]
        F = np.zeros((3, n_intf))

        F[0] = m_dot
        F[1] = np.where(m_dot >= 0.0, m_dot * uL, m_dot * uR)
        F[2] = np.where(m_dot >= 0.0, m_dot * HL, m_dot * HR)

        # Pressure flux
        p_half = _pressure_split_plus(ML) * pL + _pressure_split_minus(MR) * pR
        F[1] += p_half

        return F


class AUSMPlusUp(NumericalScheme):
    """AUSM+-up flux scheme (Liou, 2006).

    Extends AUSM+ with three corrections that improve behaviour near
    discontinuities and at low Mach numbers:

    1. **Mach dissipation** M_p: a pressure-difference term added to the
       interface Mach number, proportional to (p_R - p_L) / (rho_bar * a^2).
       This damps pressure oscillations near shocks and rarefaction edges.

    2. **Pressure dissipation** p_u: a velocity-difference term added to the
       interface pressure, proportional to rho_bar * a * (u_L - u_R).
       This adds dissipation where velocity jumps occur.

    3. **Velocity scaling** f_a = M_bar * (2 - M_bar): scales both
       dissipation terms by the local Mach level so that dissipation
       vanishes in purely supersonic flow (where upwinding is exact).

    Default parameters: Kp = 0.25, Ku = 0.75, sigma = 1.0 (Liou, 2006).

    Fully vectorised, composable with higher-order reconstructions.

    See also: docs/04_flux_schemes.md, Section 4.9.

    References
    ----------
    [Liou, 2006] M.-S. Liou, "A sequel to AUSM, Part II: AUSM+-up for
        all speeds", J. Comput. Phys., 214, pp. 137-170.
    """

    def __init__(self, Kp: float = 0.25, Ku: float = 0.75, sigma: float = 1.0):
        self._Kp = Kp
        self._Ku = Ku
        self._sigma = sigma

    @property
    def name(self) -> str:
        return "AUSM+-up"

    @property
    def scheme_type(self) -> str:
        return "FVS"

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        gamma = gas.gamma

        rhoL, uL, pL = conservative_to_primitive(UL, gamma)
        rhoR, uR, pR = conservative_to_primitive(UR, gamma)
        aL = sound_speed(rhoL, pL, gamma)
        aR = sound_speed(rhoR, pR, gamma)
        HL = (UL[2] + pL) / rhoL
        HR = (UR[2] + pR) / rhoR

        # Interface speed of sound and averages
        a_half = 0.5 * (aL + aR)
        rho_half = 0.5 * (rhoL + rhoR)

        # Left and right Mach numbers
        ML = uL / a_half
        MR = uR / a_half

        # Reference Mach for scaling: M_bar^2 = min((uL^2 + uR^2)/(2*a^2), 1)
        M_bar_sq = np.minimum((uL**2 + uR**2) / (2.0 * a_half**2), 1.0)
        M_bar = np.sqrt(M_bar_sq)

        # Velocity scaling: f_a = M_bar * (2 - M_bar)
        # f_a -> 0 at low Mach, f_a -> 1 at M_bar = 1
        fa = M_bar * (2.0 - M_bar)

        # --- Interface Mach number with pressure-based correction ---
        M_half_base = _mach_split_plus(ML) + _mach_split_minus(MR)

        # Mach dissipation: M_p = -Kp * max(1-M_bar^2, 0) * dp / (rho*a^2)
        # Liou (2006) eq. (73): the factor max(1-M_bar^2, 0) naturally
        # vanishes in supersonic flow and stays bounded as M_bar -> 0.
        #
        # Scale Mp by min(1, |dp|/p_avg) to avoid excessive correction when
        # the pressure jump is large relative to the mean pressure (as
        # happens with high-order reconstructions that produce sharper
        # interface states).
        p_avg = 0.5 * (pL + pR)
        dp = pR - pL
        scale = np.minimum(1.0, np.abs(dp) / np.maximum(p_avg, 1e-30))
        Mp = -self._Kp * scale * np.maximum(1.0 - M_bar_sq, 0.0) * dp / (rho_half * a_half**2)

        M_half = M_half_base + Mp

        # --- Mass flux ---
        m_dot = a_half * np.where(
            M_half >= 0.0,
            M_half * rhoL,
            M_half * rhoR,
        )

        # --- Convective flux ---
        n_intf = UL.shape[1]
        F = np.zeros((3, n_intf))

        F[0] = m_dot
        F[1] = np.where(m_dot >= 0.0, m_dot * uL, m_dot * uR)
        F[2] = np.where(m_dot >= 0.0, m_dot * HL, m_dot * HR)

        # --- Interface pressure with velocity-based correction ---
        p_half_base = _pressure_split_plus(ML) * pL + _pressure_split_minus(MR) * pR

        # Pressure dissipation: p_u = -Ku * P+ * P- * (rhoL+rhoR) * fa * a * du
        # fa scales the velocity dissipation — vanishes at low Mach where
        # pressure splitting alone is sufficient.
        Pp = _pressure_split_plus(ML)
        Pm = _pressure_split_minus(MR)
        pu = -self._Ku * Pp * Pm * (rhoL + rhoR) * fa * a_half * (uR - uL)

        F[1] += p_half_base + pu

        return F
