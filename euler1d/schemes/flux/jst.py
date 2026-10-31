"""Jameson-Schmidt-Turkel (JST) central scheme with adaptive artificial dissipation.

The JST scheme combines a centred flux average with two levels of artificial
dissipation that switch automatically via a pressure-based shock sensor:

* Near shocks, a second-order dissipation term (similar to Rusanov) is
  activated to suppress oscillations.
* In smooth regions, a fourth-order background dissipation (biharmonic) damps
  only the high-frequency numerical modes without degrading the second-order
  accuracy of the centred flux.

The shock sensor is a normalised second difference of the pressure field.  It
is spread over four neighbours at each interface so that the second-order
dissipation covers the entire shock vicinity.  The fourth-order dissipation is
the complement: it activates only where the sensor is small.

JST uses two tuneable parameters, kappa2 (shock dissipation intensity,
default 1.0) and kappa4 (background smoothing, default 1/64).  The default
time integrator is RK4, following the original aerodynamic application where
RK4 acts as an efficient iterative smoother.

The fourth-order term spans four cells and the sensor, spread over four
neighbours, needs one more on each side: n_ghost = 3.
JST overrides ``compute_fluxes`` directly and does not use
``compute_riemann_flux``.  It is not composable with higher-order
reconstructions.

References:
    [Jameson, Schmidt, Turkel, 1981] A. Jameson, W. Schmidt, E. Turkel,
        AIAA Paper 81-1259.

See also: docs/04_flux_schemes.md, Section 4.8.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import compute_flux, conservative_to_primitive, sound_speed
from euler1d.schemes.base import NumericalScheme


class JST(NumericalScheme):
    """Jameson-Schmidt-Turkel centred scheme with adaptive dissipation.

    A centred flux supplemented by two tiers of artificial dissipation that
    switch automatically based on a pressure-based shock sensor:

    * Second-order dissipation near shocks (Rusanov-like, suppresses
      oscillations).
    * Fourth-order background dissipation in smooth regions (biharmonic,
      damps only high-frequency numerical modes).

    The scheme overrides ``compute_fluxes`` directly; ``compute_riemann_flux``
    is not used.  Default time integrator is RK4.  Not composable with
    higher-order reconstructions.  Requires n_ghost = 3.

    Parameters
    ----------
    kappa2 : float
        Coefficient for the second-order (shock) dissipation (default 1.0).
    kappa4 : float
        Coefficient for the fourth-order (background) dissipation
        (default 1/64).

    See also: docs/04_flux_schemes.md, Section 4.8.
    """

    def __init__(self, kappa2: float = 1.0, kappa4: float = 1.0 / 64.0):
        self._kappa2 = kappa2
        self._kappa4 = kappa4

    @property
    def name(self) -> str:
        return "JST"

    @property
    def order(self) -> int:
        return 2

    @property
    def scheme_type(self) -> str:
        return "centered"

    @property
    def default_time_integrator(self) -> str:
        return "RK4"

    @property
    def n_ghost(self) -> int:
        # The sensor at interface j+1/2 uses nu at cells j-1 .. j+2, and nu at
        # a cell needs its two neighbours: three ghost cells.  With two, the
        # sensor had to be truncated at the boundary interfaces.
        return 3

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        # Not used — JST overrides compute_fluxes directly
        raise NotImplementedError("JST uses compute_fluxes, not compute_riemann_flux")

    def compute_fluxes(
        self, U: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute JST fluxes at all cell interfaces.

        Builds the centred flux average and subtracts the adaptive artificial
        dissipation.  The pressure-based shock sensor activates second-order
        dissipation near discontinuities and fourth-order background
        dissipation elsewhere.  All operations are vectorised.

        Parameters
        ----------
        U : ndarray, shape (3, N + 6)
            Conservative variables with 3 ghost cells on each side.
        gas : GasProperties
            Gas model (gamma).
        dx, dt : float
            Cell width and time step.

        Returns
        -------
        ndarray, shape (3, N + 1)
            Numerical fluxes at the N + 1 cell interfaces.

        See also: docs/04_flux_schemes.md, Section 4.8.
        """
        gamma = gas.gamma
        rho, u, p = conservative_to_primitive(U, gamma)
        a = sound_speed(rho, p, gamma)

        # Spectral radius at each cell
        lam = np.abs(u) + a  # (N+6,)

        # Physical flux at each cell
        F = compute_flux(U, gamma)  # (3, N+6)

        # Interface j+1/2 (j = 0..N) lies between extended cells j+2 and j+3.
        left = slice(2, -3)   # cells j+2, size N+1
        right = slice(3, -2)  # cells j+3, size N+1

        # --- Pressure-based shock sensor ---
        # nu_i = |p_{i+1} - 2*p_i + p_{i-1}| / (p_{i+1} + 2*p_i + p_{i-1}),
        # defined for extended cells 1..N+4 (array index i-1, size N+4)
        nu = np.abs(p[2:] - 2.0 * p[1:-1] + p[:-2]) / (p[2:] + 2.0 * p[1:-1] + p[:-2])

        # Spread over the four cells around the interface (cells j+1 .. j+4,
        # i.e. nu indices j .. j+3): eps2 = kappa2 * max(nu_{j-1}, ..., nu_{j+2})
        nu_intf = np.maximum(np.maximum(nu[0:-3], nu[1:-2]), np.maximum(nu[2:-1], nu[3:]))

        lam_half = 0.5 * (lam[left] + lam[right])  # size N+1

        eps2 = self._kappa2 * nu_intf
        eps4 = np.maximum(0.0, self._kappa4 - eps2)

        # --- Dissipation terms ---
        # d2_{j+1/2} = eps2 * lam_{j+1/2} * (U_{j+1} - U_j)
        d2 = eps2 * lam_half * (U[:, right] - U[:, left])

        # d4_{j+1/2} = eps4 * lam_{j+1/2} * (U_{j+2} - 3*U_{j+1} + 3*U_j - U_{j-1})
        dU3 = U[:, 4:-1] - 3.0 * U[:, right] + 3.0 * U[:, left] - U[:, 1:-4]
        d4 = eps4 * lam_half * dU3

        # --- Central flux + dissipation ---
        F_half = 0.5 * (F[:, left] + F[:, right]) - d2 + d4

        return F_half
