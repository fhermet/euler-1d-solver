"""Common base class for reconstruction + Riemann-flux schemes.

Each reconstruction (MUSCL, ENO, WENO3, WENO5) only supplies its 1D formula,
the ``kernel``; the choice of reconstructed variables and the first-order
fallback on unphysical faces are handled once, in
:func:`euler1d.schemes.reconstruction.variables.reconstruct_faces`.

See docs/03_finite_volume.md, section 3.5 (composition principle).
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from euler1d.config import GasProperties
from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.reconstruction.variables import check_variables, reconstruct_faces


class ReconstructedScheme(NumericalScheme):
    """Reconstruction of interface states followed by a Riemann flux.

    Parameters
    ----------
    riemann_solver : NumericalScheme
        The underlying first-order flux solver.
    variables : str, default "primitive"
        Variables reconstructed component by component: "primitive",
        "conservative" or "characteristic".
    """

    def __init__(self, riemann_solver: NumericalScheme, variables: str = "primitive"):
        self._riemann = riemann_solver
        self._variables = check_variables(variables)

    @property
    @abstractmethod
    def label(self) -> str:
        """Reconstruction name used to build ``name``, e.g. ``"WENO5"``."""

    @abstractmethod
    def kernel(self, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
        """1D formula: stencils (2*n_ghost, n_var, n_intf) -> (left, right) faces."""

    @property
    def name(self) -> str:
        suffix = "" if self._variables == "primitive" else f" [{self._variables}]"
        return f"{self.label}-{self._riemann.name}{suffix}"

    def reconstruct(
        self, U: np.ndarray, gamma: float, dx: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Left and right conservative states at the interfaces of U."""
        return reconstruct_faces(U, gamma, dx, self.kernel, self.n_ghost, self._variables)

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        return self._riemann.compute_riemann_flux(UL, UR, gas, dx, dt)

    def compute_fluxes(
        self, U: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        UL, UR = self.reconstruct(U, gas.gamma, dx)
        return self.compute_riemann_flux(UL, UR, gas, dx, dt)
