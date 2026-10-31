"""Fourier analysis of numerical dissipation and dispersion.

This module quantifies two fundamental error sources of finite-volume schemes
in smooth regions: numerical dissipation (amplitude damping) and numerical
dispersion (phase/wave-speed error).  The analysis is performed by linearising
the 1-D Euler equations around a uniform base state (rho0, u0, p0) = (1, 1, 1)
and measuring the complex amplification factor G(theta) for each reduced
wavenumber theta = k*dx.

Methodology overview
--------------------
1. Select an eigenmode of the linearised primitive-variable system (entropy
   wave at speed u0, right-going acoustic wave at u0 + c0, or left-going
   acoustic wave at u0 - c0).
2. For each discrete wavenumber, inject two perturbations (cosine and sine)
   aligned with the chosen eigenvector.
3. Advance each perturbation by one **complete** time step (spatial operator
   *and* Runge-Kutta stages) so that the measured G includes the effect of
   the time integrator.
4. Project the output onto the corresponding left eigenvector to isolate the
   excited mode from the 3-component conservative/primitive system.
5. Extract G_real and G_imag via discrete Fourier projection, then compute
   |G| (dissipation) and phase ratio phi_num / phi_exact (dispersion).

The dual-perturbation technique (steps 2 and 5) is necessary because a single
cosine simulation cannot separate the real and imaginary parts of G.  Full
details are given in docs/07_fourier_analysis.md, sections 7.3--7.4.

What the analysis does and does not measure
-------------------------------------------
* The perturbation amplitude is 1e-6.  WENO smoothness indicators are then
  far below the regularisation eps, so the weights equal their ideal values:
  the curves are those of the underlying linear scheme, i.e. WENO in smooth
  regions.
* Slope limiters are homogeneous of degree one (they depend on ratios of
  differences), so MUSCL results do not depend on the amplitude, but the
  limiter clips the cosine at its extrema: G is that of the fundamental only.
* |G| <= 1 means linear (von Neumann) stability.  It does not imply the TVD
  property: MUSCL at CFL 0.9 has |G| <= 1 here, yet it is not TVD and lets
  round-off grow on the Shu-Osher case (docs/06_time_integration.md, 6.1).

Reference: [Toro, 2009] Ch. 13 -- Fourier analysis of FV schemes.

See also
--------
docs/07_fourier_analysis.md : Chapter 7 of the project documentation.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import conservative_to_primitive, primitive_to_conservative
from euler1d.schemes.base import NumericalScheme
from euler1d.solver import _compute_rhs, _rk_step

WAVE_TYPES = ("entropy", "acoustic", "acoustic-left")


def _single_step(U: np.ndarray, scheme: NumericalScheme, gas: GasProperties,
                 dx: float, dt: float, rk: str) -> np.ndarray:
    """Advance *U* by one full time step on a periodic domain.

    The complete Runge-Kutta step is taken, not only the spatial operator:
    the amplification factor must include the time integrator, which changes
    both |G| and the phase.  The same ``_rk_step`` as the solver is used.
    """
    return _rk_step(U, rk, lambda v: _compute_rhs(v, scheme, gas, dx, dt, "periodic"))


def compute_amplification(
    scheme: NumericalScheme,
    cfl: float = 0.5,
    n_cells: int = 256,
    gamma: float = 1.4,
    n_modes: int = 128,
    wave_type: str = "entropy",
    time_integrator: str | None = None,
    respect_cfl_limit: bool = True,
) -> dict:
    """Compute the numerical amplification factor G(theta) for a scheme.

    For each reduced wavenumber theta = k*dx, this function measures the complex
    amplification factor G(theta) that encodes both numerical dissipation
    (|G| < 1 means damping) and numerical dispersion (arg(G) != exact phase).

    The analysis linearises the Euler equations around (rho0, u0, p0) = (1, 1, 1)
    on a periodic domain and uses a **dual-perturbation technique** (see
    docs/07_fourier_analysis.md, sections 7.3--7.4):

    1. For each mode *m*, inject a cosine perturbation eps * eigenvector * cos(kx)
       and a sine perturbation eps * eigenvector * sin(kx), where the eigenvector
       is aligned with the selected wave family.
    2. Advance each perturbation by one complete time step (spatial operator +
       full RK integration).
    3. Convert back to primitive variables and compute the output perturbation
       delta = W_out - W0.
    4. **Project** delta onto the excited eigenmode using the corresponding left
       eigenvector of the primitive-variable Jacobian.  Left eigenvectors used:
       - Entropy: l = (1, 0, -1/c0^2)
       - Right acoustic: l = (0, rho0/(2*c0), 1/(2*c0^2))
       - Left acoustic: l = (0, -rho0/(2*c0), 1/(2*c0^2))
    5. Extract Fourier coefficients a_cc, a_cs, a_sc, a_ss by discrete dot
       products with cos(kx) and sin(kx), then reconstruct:
       G_real = (a_cc + a_ss) / (2*A0),  G_imag = (a_sc - a_cs) / (2*A0).
    6. Compute |G| and phase ratio phi_num / phi_exact.

    Parameters
    ----------
    scheme : NumericalScheme
        Numerical scheme to analyse.
    cfl : float, optional
        Requested CFL number; dt = cfl * dx / (|u0| + c0).  Default 0.5.
    n_cells : int, optional
        Number of cells in the periodic domain.  Default 256.
    gamma : float, optional
        Heat capacity ratio.  Default 1.4.
    n_modes : int, optional
        Number of Fourier modes to analyse, from m = 1 to m = n_modes.
        Default 128 (up to theta = pi for 256 cells).
    wave_type : str, optional
        Wave family to excite:
        - ``"entropy"`` -- density perturbation advected at u0,
          eigenvector (drho, du, dp) = (1, 0, 0).
        - ``"acoustic"`` -- right-going acoustic wave at u0 + c0,
          eigenvector (1, c0/rho0, c0^2).
        - ``"acoustic-left"`` -- left-going acoustic wave at u0 - c0 (here
          about -0.18: a nearly sonic wave), eigenvector (1, -c0/rho0, c0^2).
        Default ``"entropy"``.
    time_integrator : str or None, optional
        RK method ("RK1" ... "RK5"); None uses the scheme's default.  Schemes
        with built-in time integration (Lax-Wendroff) always use RK1.
    respect_cfl_limit : bool, optional
        If True (default), cap the CFL by ``scheme.max_cfl`` as the solver
        does.  Set to False to explore stability beyond the guaranteed limit.

    Returns
    -------
    dict
        - **theta** : reduced wavenumbers 2*pi*m / n_cells.
        - **abs_G** : modulus |G(theta)|; > 1 means a growing mode.
        - **phase_ratio** : phi_num / phi_exact (1 for a perfect scheme).
          NaN at theta = pi, where the grid mode is a standing sawtooth with
          no phase, and where |G| < 1e-4, below which the measured phase is
          dominated by the linearisation error (about 1e-6 on G).
        - **cfl** : CFL number actually used.
        - **time_integrator** : RK method actually used.
    """
    if wave_type not in WAVE_TYPES:
        raise ValueError(f"Unknown wave type {wave_type!r}; expected one of {WAVE_TYPES}")
    rho0, u0, p0 = 1.0, 1.0, 1.0
    gas = GasProperties(gamma=gamma)
    c0 = np.sqrt(gamma * p0 / rho0)

    if respect_cfl_limit:
        cfl = min(cfl, scheme.max_cfl)
    if scheme.time_integral_included:
        rk = "RK1"
    else:
        rk = time_integrator or scheme.default_time_integrator

    L = 1.0
    dx = L / n_cells
    x = np.linspace(0.5 * dx, L - 0.5 * dx, n_cells)
    dt = cfl * dx / (abs(u0) + c0)

    # Right eigenvector r (normalised so that drho = 1), wave speed, and left
    # eigenvector l with l . r = 1, used to project the output on the mode.
    if wave_type == "acoustic":
        r_vec = (1.0, c0 / rho0, c0**2)
        l_vec = (0.0, rho0 / (2.0 * c0), 1.0 / (2.0 * c0**2))
        wave_speed = u0 + c0
    elif wave_type == "acoustic-left":
        r_vec = (1.0, -c0 / rho0, c0**2)
        l_vec = (0.0, -rho0 / (2.0 * c0), 1.0 / (2.0 * c0**2))
        wave_speed = u0 - c0
    else:
        r_vec = (1.0, 0.0, 0.0)
        l_vec = (1.0, 0.0, -1.0 / c0**2)
        wave_speed = u0

    sigma = wave_speed * dt / dx
    eps = 1e-6

    def response(shape):
        """Projected output perturbation for an input eps * r * shape(x)."""
        U = primitive_to_conservative(rho0 + eps * r_vec[0] * shape,
                                      u0 + eps * r_vec[1] * shape,
                                      p0 + eps * r_vec[2] * shape, gamma)
        rho, u, p = conservative_to_primitive(_single_step(U, scheme, gas, dx, dt, rk), gamma)
        return l_vec[0] * (rho - rho0) + l_vec[1] * (u - u0) + l_vec[2] * (p - p0)

    modes = np.arange(1, n_modes + 1)
    theta = 2.0 * np.pi * modes / n_cells

    abs_G = np.zeros(n_modes)
    phase_G = np.zeros(n_modes)

    for idx, m in enumerate(modes):
        k = 2.0 * np.pi * m / L
        cos_k = np.cos(k * x)
        sin_k = np.sin(k * x)
        delta_c = response(cos_k)
        delta_s = response(sin_k)

        norm = 2.0 / n_cells
        a_cc = norm * np.sum(delta_c * cos_k)
        a_cs = norm * np.sum(delta_c * sin_k)
        a_sc = norm * np.sum(delta_s * cos_k)
        a_ss = norm * np.sum(delta_s * sin_k)

        G_real = 0.5 * (a_cc + a_ss) / eps
        G_imag = 0.5 * (a_sc - a_cs) / eps

        abs_G[idx] = np.hypot(G_real, G_imag)
        phase_G[idx] = np.arctan2(G_imag, G_real)

    phase_G = np.unwrap(phase_G)
    exact_phase = -sigma * theta

    nyquist = 2 * modes == n_cells
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_ratio = np.where(
            (np.abs(exact_phase) > 1e-12) & (abs_G > 1e-4) & ~nyquist,
            phase_G / exact_phase,
            np.nan,
        )

    return {
        "theta": theta,
        "abs_G": abs_G,
        "phase_ratio": phase_ratio,
        "cfl": cfl,
        "time_integrator": rk,
    }
