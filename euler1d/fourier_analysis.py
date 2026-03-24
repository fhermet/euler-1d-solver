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
   wave at speed u0, or right-going acoustic wave at speed u0 + c0).
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
from euler1d.solver import _compute_rhs


def _single_step(U: np.ndarray, scheme: NumericalScheme, gas: GasProperties,
                 dx: float, dt: float) -> np.ndarray:
    """Advance *U* by one full time step using the scheme's default integrator.

    This function reproduces the complete Runge-Kutta integration (not just the
    spatial operator) because the Fourier analysis must measure the combined
    behaviour of spatial discretisation **and** time integration.  The time
    integrator affects both |G| and the numerical phase: for instance, RK1
    introduces first-order temporal dissipation, while SSP-RK3 preserves the
    TVD property of the spatial reconstruction.

    The integrator is selected automatically from ``scheme.default_time_integrator``
    (or forced to RK1 when ``scheme.time_integral_included`` is True, e.g.
    Lax-Wendroff).  Five methods are supported:

    * **RK1** -- Forward Euler (1 stage).  Used by first-order flux schemes and
      Lax-Wendroff (which already embeds temporal integration).
    * **RK2** -- Heun's method (2 stages).  Default for MUSCL and ENO2 schemes.
    * **RK3** -- SSP-RK3 / Shu-Osher (3 stages).  Default for WENO3.  Strong
      stability preserving.
    * **RK4** -- Classical RK4 (4 stages).  Default for JST.
    * **RK5** -- Dormand-Prince (6 stages, 5th-order accurate).  Default for
      WENO5.

    Parameters
    ----------
    U : np.ndarray, shape (3, N)
        Conservative variables on the periodic domain.
    scheme : NumericalScheme
        The numerical scheme whose spatial operator and default RK method are
        used.
    gas : GasProperties
        Gas model (provides gamma).
    dx : float
        Cell width.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray, shape (3, N)
        Conservative variables after one time step.
    """
    bc = "periodic"
    rk = "RK1" if scheme.time_integral_included else scheme.default_time_integrator

    k1 = _compute_rhs(U, scheme, gas, dx, dt, bc)
    if rk == "RK1":
        return U + k1
    elif rk == "RK2":
        k2 = _compute_rhs(U + k1, scheme, gas, dx, dt, bc)
        return U + 0.5 * (k1 + k2)
    elif rk == "RK3":
        # SSP-RK3 (Shu-Osher)
        U1 = U + k1
        k2 = _compute_rhs(U1, scheme, gas, dx, dt, bc)
        U2 = 0.75 * U + 0.25 * (U1 + k2)
        k3 = _compute_rhs(U2, scheme, gas, dx, dt, bc)
        return (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + k3)
    elif rk == "RK4":
        k2 = _compute_rhs(U + 0.5 * k1, scheme, gas, dx, dt, bc)
        k3 = _compute_rhs(U + 0.5 * k2, scheme, gas, dx, dt, bc)
        k4 = _compute_rhs(U + k3, scheme, gas, dx, dt, bc)
        return U + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    else:  # RK5 — Dormand-Prince
        rhs = lambda v: _compute_rhs(v, scheme, gas, dx, dt, bc)
        k2 = rhs(U + (1.0/5.0) * k1)
        k3 = rhs(U + (3.0/40.0) * k1 + (9.0/40.0) * k2)
        k4 = rhs(U + (44.0/45.0) * k1 - (56.0/15.0) * k2 + (32.0/9.0) * k3)
        k5 = rhs(U + (19372.0/6561.0) * k1 - (25360.0/2187.0) * k2
                 + (64448.0/6561.0) * k3 - (212.0/729.0) * k4)
        k6 = rhs(U + (9017.0/3168.0) * k1 - (355.0/33.0) * k2
                 + (46732.0/5247.0) * k3 + (49.0/176.0) * k4
                 - (5103.0/18656.0) * k5)
        return U + (35.0/384.0) * k1 + (500.0/1113.0) * k3 \
                 + (125.0/192.0) * k4 - (2187.0/6784.0) * k5 \
                 + (11.0/84.0) * k6


WAVE_TYPES = ("entropie", "acoustique")


def compute_amplification(
    scheme: NumericalScheme,
    cfl: float = 0.5,
    n_cells: int = 256,
    gamma: float = 1.4,
    n_modes: int = 128,
    wave_type: str = "entropie",
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
       is aligned with the selected wave family (entropy or right acoustic).
    2. Advance each perturbation by one complete time step via ``_single_step``
       (spatial operator + full RK integration).
    3. Convert back to primitive variables and compute the output perturbation
       delta = W_out - W0.
    4. **Project** delta onto the excited eigenmode using the corresponding left
       eigenvector of the primitive-variable Jacobian A0.  This isolates the
       scalar amplitude of the excited mode and filters out cross-mode
       contamination.  Left eigenvectors used:
       - Entropy: l = (1, 0, -1/c0^2)
       - Right acoustic: l = (0, rho0/(2*c0), 1/(2*c0^2))
    5. Extract Fourier coefficients a_cc, a_cs, a_sc, a_ss by discrete dot
       products with cos(kx) and sin(kx), then reconstruct:
       G_real = (a_cc + a_ss) / (2*A0),  G_imag = (a_sc - a_cs) / (2*A0).
    6. Compute |G| and phase ratio phi_num / phi_exact.

    Two simulations (cos and sin) are needed because a single perturbation only
    yields two of the four coefficients, which is insufficient to separate the
    real and imaginary parts of G.

    Parameters
    ----------
    scheme : NumericalScheme
        Numerical scheme to analyse (includes spatial operator and default RK).
    cfl : float, optional
        CFL number used to compute dt = cfl * dx / (|u0| + c0).  Default 0.5.
    n_cells : int, optional
        Number of cells in the periodic domain.  Default 256.
    gamma : float, optional
        Heat capacity ratio.  Default 1.4.
    n_modes : int, optional
        Number of Fourier modes to analyse, from m = 1 to m = n_modes.
        Default 128.
    wave_type : str, optional
        Wave family to excite:
        - ``"entropie"`` -- density-only perturbation advected at u0.
          Eigenvector (drho, du, dp) = (1, 0, 0).
        - ``"acoustique"`` -- right-going acoustic perturbation at u0 + c0.
          Eigenvector (drho, du, dp) = (1, c0/rho0, c0^2).
        Default ``"entropie"``.

    Returns
    -------
    dict
        Dictionary with three numpy arrays:

        - **theta** : ndarray, shape (n_modes,)
          Reduced wavenumbers theta = 2*pi*m / n_cells, from near 0 to near pi.
        - **abs_G** : ndarray, shape (n_modes,)
          Modulus |G(theta)|.  Values < 1 indicate numerical dissipation;
          values > 1 indicate instability.
        - **phase_ratio** : ndarray, shape (n_modes,)
          Ratio phi_num / phi_exact.  Equals 1 for a perfect scheme.  Set to
          NaN when the mode is too damped (|G| < 1e-10) or the exact phase is
          near zero (|phi_exact| < 1e-12), since the phase ratio is then
          physically meaningless.
    """
    rho0, u0, p0 = 1.0, 1.0, 1.0
    gas = GasProperties(gamma=gamma)
    c0 = np.sqrt(gamma * p0 / rho0)

    L = 1.0
    dx = L / n_cells
    x = np.linspace(0.5 * dx, L - 0.5 * dx, n_cells)

    s_max = abs(u0) + c0
    dt = cfl * dx / s_max

    # Perturbation eigenvector and wave speed
    if wave_type == "acoustique":
        # Right-going acoustic: eigenvector (drho, du, dp) = (rho0, c0, rho0*c0^2)
        # normalized so drho = 1
        evec_rho = 1.0
        evec_u = c0 / rho0
        evec_p = c0**2
        wave_speed = u0 + c0
    else:
        # Entropy wave: eigenvector (drho, du, dp) = (1, 0, 0)
        evec_rho = 1.0
        evec_u = 0.0
        evec_p = 0.0
        wave_speed = u0

    sigma = wave_speed * dt / dx
    eps = 1e-6

    # Left eigenvectors of the primitive-variable Jacobian (rows of R^{-1})
    # for projecting the output perturbation onto the excited mode.
    #   Entropy:          l = (1, 0, -1/c0^2)
    #   Right acoustic:   l = (0, rho0/(2*c0), 1/(2*c0^2))
    if wave_type == "acoustique":
        def _project(drho, du, dp):
            return rho0 / (2.0 * c0) * du + dp / (2.0 * c0**2)
        # l_plus · perturbation = l_plus · eps*(1, c0/rho0, c0^2)
        #   = eps * (rho0/(2*c0) * c0/rho0 + c0^2/(2*c0^2))
        #   = eps * (1/2 + 1/2) = eps
        input_amplitude = eps
    else:
        def _project(drho, du, dp):
            return drho - dp / c0**2
        # l_entropy · eps*(1,0,0) = eps
        input_amplitude = eps

    modes = np.arange(1, n_modes + 1)
    theta = 2.0 * np.pi * modes / n_cells

    abs_G = np.zeros(n_modes)
    phase_G = np.zeros(n_modes)

    for idx, m in enumerate(modes):
        k = 2.0 * np.pi * m / L
        cos_k = np.cos(k * x)
        sin_k = np.sin(k * x)

        # --- Cosine perturbation ---
        rho_c = rho0 + eps * evec_rho * cos_k
        u_c = u0 + eps * evec_u * cos_k
        p_c = p0 + eps * evec_p * cos_k
        U_c = primitive_to_conservative(rho_c, u_c, p_c, gamma)
        U_c_new = _single_step(U_c, scheme, gas, dx, dt)
        rho_c_out, u_c_out, p_c_out = conservative_to_primitive(U_c_new, gamma)
        delta_c = _project(rho_c_out - rho0, u_c_out - u0, p_c_out - p0)

        # --- Sine perturbation ---
        rho_s = rho0 + eps * evec_rho * sin_k
        u_s = u0 + eps * evec_u * sin_k
        p_s = p0 + eps * evec_p * sin_k
        U_s = primitive_to_conservative(rho_s, u_s, p_s, gamma)
        U_s_new = _single_step(U_s, scheme, gas, dx, dt)
        rho_s_out, u_s_out, p_s_out = conservative_to_primitive(U_s_new, gamma)
        delta_s = _project(rho_s_out - rho0, u_s_out - u0, p_s_out - p0)

        # Extract amplification factor via modal projection
        norm = 2.0 / n_cells
        a_cc = norm * np.sum(delta_c * cos_k)
        a_cs = norm * np.sum(delta_c * sin_k)
        a_sc = norm * np.sum(delta_s * cos_k)
        a_ss = norm * np.sum(delta_s * sin_k)

        G_real = 0.5 * (a_cc + a_ss) / input_amplitude
        G_imag = 0.5 * (a_sc - a_cs) / input_amplitude

        abs_G[idx] = np.sqrt(G_real**2 + G_imag**2)
        phase_G[idx] = np.arctan2(G_imag, G_real)

    phase_G = np.unwrap(phase_G)
    exact_phase = -sigma * theta

    with np.errstate(divide="ignore", invalid="ignore"):
        phase_ratio = np.where(
            (np.abs(exact_phase) > 1e-12) & (abs_G > 1e-10),
            phase_G / exact_phase,
            np.nan,
        )

    return {
        "theta": theta,
        "abs_G": abs_G,
        "phase_ratio": phase_ratio,
    }
