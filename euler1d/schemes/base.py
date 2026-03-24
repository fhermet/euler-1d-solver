"""Abstract base class for numerical schemes.

Every finite volume scheme in this project inherits from
:class:`NumericalScheme`.  The class encodes the **composition principle**
described in docs/03_finite_volume.md, §3.5: a scheme is the combination of a
spatial *reconstruction* (from cell averages to interface states) and a
*numerical flux* (approximate Riemann solver that turns left/right states into
a single interface flux).

The default implementation of :meth:`compute_fluxes` performs a first-order
piecewise-constant reconstruction (U_L = U_i, U_R = U_{i+1}) and delegates to
the abstract :meth:`compute_riemann_flux`.  Higher-order schemes override
:meth:`compute_fluxes` to insert their own reconstruction step (MUSCL, ENO,
WENO) before calling :meth:`compute_riemann_flux`.

Centred schemes (Lax-Friedrichs, Lax-Wendroff, JST) do not decompose into
reconstruction + Riemann solver; they override both methods as needed.

Key properties exposed by each scheme:

* ``order`` -- formal spatial order of accuracy (1 through 5).
* ``n_ghost`` -- number of ghost cells required on each side, determined by
  the stencil width of the reconstruction.
* ``time_integral_included`` -- whether the flux already embeds temporal
  integration (e.g. Lax-Wendroff), in which case the time integrator must
  use a single stage (RK1).
* ``default_time_integrator`` -- the Runge-Kutta method automatically
  selected to match the spatial order when the user chooses "Auto".

See also: docs/03_finite_volume.md, §3.5 (composition principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from euler1d.config import GasProperties


class NumericalScheme(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def order(self) -> int:
        return 1

    @property
    def scheme_type(self) -> str:
        """Return 'upwind' or 'centré'."""
        return "upwind"

    @property
    def time_integral_included(self) -> bool:
        """True if the flux already includes temporal integration (e.g. Lax-Wendroff)."""
        return False

    @property
    def default_time_integrator(self) -> str:
        """Default RK method when user selects 'Auto'."""
        if self.order < 2:
            return "RK1"
        if self.order < 3:
            return "RK2"
        if self.order < 4:
            return "RK3"
        if self.order < 5:
            return "RK4"
        return "RK5"

    @property
    def n_ghost(self) -> int:
        return 1

    @abstractmethod
    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute numerical flux at interfaces given left/right states.

        Parameters
        ----------
        UL, UR : array (3, N_intf)
            Left and right conservative states at each interface.
        gas : GasProperties
        dx, dt : float

        Returns
        -------
        F : array (3, N_intf)
            Numerical flux at each interface.
        """

    def compute_fluxes(
        self, U: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        """Compute numerical fluxes at cell interfaces.

        Parameters
        ----------
        U : array (3, N + 2*n_ghost)
            Conservative variables with ghost cells.
        gas : GasProperties
        dx, dt : float

        Returns
        -------
        F : array (3, N+1)
            Fluxes at the N+1 interfaces.
        """
        UL = U[:, :-1]
        UR = U[:, 1:]
        return self.compute_riemann_flux(UL, UR, gas, dx, dt)
