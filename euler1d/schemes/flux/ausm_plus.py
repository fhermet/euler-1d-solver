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
    - Scaling function f_a: adapts both terms, and the pressure splitting, to
      the local Mach regime, with a cut-off Mach number M_inf.

Both use Liou's interface sound speed, built from the critical sound speed
a*^2 = 2(gamma-1)/(gamma+1) H rather than an arithmetic average.

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
from euler1d.physics import conservative_to_primitive
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


def _pressure_split_plus(M: np.ndarray, alpha: np.ndarray | float = 3.0 / 16.0) -> np.ndarray:
    """P+ splitting for the pressure flux (alpha = 3/16 in AUSM+)."""
    return np.where(
        np.abs(M) >= 1.0,
        0.5 * (1.0 + np.sign(M)),
        0.25 * (M + 1.0)**2 * (2.0 - M) + alpha * M * (M**2 - 1.0)**2,
    )


def _pressure_split_minus(M: np.ndarray, alpha: np.ndarray | float = 3.0 / 16.0) -> np.ndarray:
    """P- splitting for the pressure flux (alpha = 3/16 in AUSM+)."""
    return np.where(
        np.abs(M) >= 1.0,
        0.5 * (1.0 - np.sign(M)),
        0.25 * (M - 1.0)**2 * (2.0 + M) - alpha * M * (M**2 - 1.0)**2,
    )


def _interface_sound_speed(
    uL: np.ndarray, HL: np.ndarray, uR: np.ndarray, HR: np.ndarray, gamma: float
) -> np.ndarray:
    """Interface sound speed of Liou (1996, eq. 26; 2006, eq. 30).

    a_1/2 = min(a~_L, a~_R), with a~_L = a*_L^2 / max(a*_L, u_L),
    a~_R = a*_R^2 / max(a*_R, -u_R) and a*^2 = 2(gamma-1)/(gamma+1) H.
    """
    a_star_sq_L = 2.0 * (gamma - 1.0) / (gamma + 1.0) * HL
    a_star_sq_R = 2.0 * (gamma - 1.0) / (gamma + 1.0) * HR
    aL = a_star_sq_L / np.maximum(np.sqrt(a_star_sq_L), uL)
    aR = a_star_sq_R / np.maximum(np.sqrt(a_star_sq_R), -uR)
    return np.minimum(aL, aR)


class AUSMPlus(NumericalScheme):
    """AUSM+ flux scheme (Liou, 1996).

    Splits the interface flux into a convective part (mass flux times
    transported quantities) and a pressure part, each using smooth
    polynomial splittings of the local Mach number.

    The interface speed of sound a_{1/2} is Liou's (see
    ``_interface_sound_speed``), and the interface Mach number is
    M_{1/2} = M+_L + M-_R where M_L = u_L / a_{1/2}, M_R = u_R / a_{1/2}.

    The mass flux carries no pressure diffusion: at a pressure jump at rest
    (u = 0, as at the start of the Sod problem) it vanishes, and the
    low-pressure cell receives momentum but no mass.  The first step then
    needs a CFL below about 0.7 on Sod.  AUSM+-up adds the missing term.

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
        HL = (UL[2] + pL) / rhoL  # total enthalpy
        HR = (UR[2] + pR) / rhoR

        a_half = _interface_sound_speed(uL, HL, uR, HR, gamma)

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

    Extends AUSM+ with corrections that improve behaviour near
    discontinuities and at low Mach numbers (Liou 2006, eqs. 70-76):

    1. **Mach dissipation** M_p, added to the interface Mach number:
       M_p = -(Kp / f_a) max(1 - sigma M_bar^2, 0) (p_R - p_L) / (rho_1/2 a_1/2^2).
       It gives the mass flux the pressure diffusion that AUSM+ lacks.

    2. **Pressure dissipation** p_u, added to the interface pressure:
       p_u = -Ku P+ P- (rho_L + rho_R) (f_a a_1/2) (u_R - u_L).

    3. **Scaling function** f_a = M_o (2 - M_o), with
       M_o^2 = min(1, max(M_bar^2, M_inf^2)) and
       M_bar^2 = (u_L^2 + u_R^2) / (2 a_1/2^2).  The pressure splitting uses
       alpha = 3/16 (-4 + 5 f_a^2).

    The cut-off M_inf keeps f_a away from zero, where Kp / f_a would make the
    explicit scheme stiff.  M_inf = 1 gives f_a = 1 (no low-Mach scaling);
    the total dissipation then exceeds what forward Euler tolerates at CFL 0.9
    and the acoustic wave case diverges.  M_inf = 0.1 makes Kp / f_a too
    large and the Sod case diverges.  The default 0.5 runs every test case up
    to CFL 0.8 (and all but the acoustic wave at CFL 1).

    Default parameters: Kp = 0.25, Ku = 0.75, sigma = 1.0 (Liou, 2006).

    Fully vectorised, composable with higher-order reconstructions.

    See also: docs/04_flux_schemes.md, Section 4.9.

    References
    ----------
    [Liou, 2006] M.-S. Liou, "A sequel to AUSM, Part II: AUSM+-up for
        all speeds", J. Comput. Phys., 214, pp. 137-170.
    """

    def __init__(
        self, Kp: float = 0.25, Ku: float = 0.75, sigma: float = 1.0, M_inf: float = 0.5
    ):
        self._Kp = Kp
        self._Ku = Ku
        self._sigma = sigma
        self._M_inf = M_inf

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
        HL = (UL[2] + pL) / rhoL
        HR = (UR[2] + pR) / rhoR

        a_half = _interface_sound_speed(uL, HL, uR, HR, gamma)
        rho_half = 0.5 * (rhoL + rhoR)

        ML = uL / a_half
        MR = uR / a_half

        # Scaling function with cut-off M_inf
        M_bar_sq = (uL**2 + uR**2) / (2.0 * a_half**2)
        M_o = np.sqrt(np.minimum(1.0, np.maximum(M_bar_sq, self._M_inf**2)))
        fa = M_o * (2.0 - M_o)
        alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa**2)

        # --- Interface Mach number with pressure diffusion ---
        Mp = (
            -self._Kp / fa
            * np.maximum(1.0 - self._sigma * M_bar_sq, 0.0)
            * (pR - pL) / (rho_half * a_half**2)
        )
        M_half = _mach_split_plus(ML) + _mach_split_minus(MR) + Mp

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

        # --- Interface pressure with velocity diffusion ---
        Pp = _pressure_split_plus(ML, alpha)
        Pm = _pressure_split_minus(MR, alpha)
        pu = -self._Ku * Pp * Pm * (rhoL + rhoR) * fa * a_half * (uR - uL)

        F[1] += Pp * pL + Pm * pR + pu

        return F
