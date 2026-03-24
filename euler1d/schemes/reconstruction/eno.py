"""ENO (Essentially Non-Oscillatory) reconstruction scheme, order 2.

Implements the ENO methodology of adaptive stencil selection: for each cell,
the reconstruction chooses the stencil with the smallest second divided
differences (i.e., the smoothest local interpolation). This avoids interpolating
across discontinuities without requiring a tunable limiter function.

The reconstruction operates on primitive variables W = (rho, u, p) and applies
a positivity floor to density and pressure.

Compared to MUSCL, ENO requires no parameter tuning. Compared to WENO, the
stencil selection is binary (abrupt switching) rather than a smooth blending,
which can introduce small perturbations near the switching threshold.

See docs/05_reconstruction.md, section 5.2 for derivation and analysis.

References
----------
- [Harten et al., 1987] A. Harten, B. Engquist, S. Osher, S. R. Chakravarthy,
  "Uniformly High Order Accurate Essentially Non-Oscillatory Schemes, III",
  J. Comput. Phys., 71, pp. 231-303.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import conservative_to_primitive, primitive_to_conservative
from euler1d.schemes.base import NumericalScheme


def eno_reconstruct(
    U: np.ndarray, gamma: float, eno_order: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Perform ENO reconstruction on primitive variables with adaptive stencil selection.

    For ENO2, computes first and second divided differences, then selects for
    each cell and each interface the stencil (left-biased or right-biased)
    whose second difference is smallest in absolute value. This criterion
    identifies the smoothest local interpolation and avoids crossing
    discontinuities. Density and pressure are clamped to 1e-10 for positivity.

    Parameters
    ----------
    U : ndarray, shape (3, N + 4)
        Conservative variables with 2 ghost cells on each side.
    gamma : float
        Ratio of specific heats.
    eno_order : int, default 2
        ENO reconstruction order (1 for piecewise-constant fallback, 2 for
        piecewise-linear with stencil selection).

    Returns
    -------
    UL : ndarray, shape (3, N + 1)
        Reconstructed left states at each interface.
    UR : ndarray, shape (3, N + 1)
        Reconstructed right states at each interface.

    Raises
    ------
    ValueError
        If eno_order is not 1 or 2.

    See Also
    --------
    ENOScheme : Scheme class that pairs this reconstruction with a Riemann solver.
    """
    rho, u, p = conservative_to_primitive(U, gamma)
    W = np.array([rho, u, p])

    if eno_order == 1:
        W_L = W[:, 1:-2]
        W_R = W[:, 2:-1]
    elif eno_order == 2:
        D = np.diff(W, axis=1)
        DD = np.diff(D, axis=1)

        n_intf = W.shape[1] - 3

        big = np.full((3, 1), 1e30)
        DD_ext = np.concatenate([big, DD, big], axis=1)

        D_left_L = D[:, 0:n_intf]
        D_right_L = D[:, 1:n_intf+1]
        use_left_L = np.abs(DD_ext[:, 1:n_intf+1]) <= np.abs(DD_ext[:, 2:n_intf+2])
        slope_L = np.where(use_left_L, D_left_L, D_right_L)
        W_L = W[:, 1:n_intf+1] + 0.5 * slope_L

        D_left_R = D[:, 1:n_intf+1]
        D_right_R = D[:, 2:n_intf+2]
        use_left_R = np.abs(DD_ext[:, 2:n_intf+2]) <= np.abs(DD_ext[:, 3:n_intf+3])
        slope_R = np.where(use_left_R, D_left_R, D_right_R)
        W_R = W[:, 2:n_intf+2] - 0.5 * slope_R
    else:
        raise ValueError(f"ENO order {eno_order} not supported (use 1 or 2)")

    W_L[0] = np.maximum(W_L[0], 1e-10)
    W_L[2] = np.maximum(W_L[2], 1e-10)
    W_R[0] = np.maximum(W_R[0], 1e-10)
    W_R[2] = np.maximum(W_R[2], 1e-10)

    UL = primitive_to_conservative(W_L[0], W_L[1], W_L[2], gamma)
    UR = primitive_to_conservative(W_R[0], W_R[1], W_R[2], gamma)

    return UL, UR


class ENOScheme(NumericalScheme):
    """ENO scheme combining adaptive stencil-selection reconstruction with a Riemann solver.

    Uses the ENO methodology to reconstruct interface states, then delegates
    flux computation to the underlying Riemann solver. The stencil selection
    is parameter-free (no limiter choice).

    Properties: order=2 (for eno_order=2), n_ghost=2, default time integrator=RK2.

    Parameters
    ----------
    riemann_solver : NumericalScheme
        The underlying first-order Riemann flux solver.
    eno_order : int, default 2
        ENO reconstruction order.

    See Also
    --------
    eno_reconstruct : The reconstruction function used internally.
    make_eno_scheme : Factory function to create an ENOScheme by solver name.
    """

    def __init__(self, riemann_solver: NumericalScheme, eno_order: int = 2):
        self._riemann = riemann_solver
        self._eno_order = eno_order

    @property
    def name(self) -> str:
        return f"ENO{self._eno_order}-{self._riemann.name}"

    @property
    def order(self) -> int:
        return self._eno_order

    @property
    def n_ghost(self) -> int:
        return 2

    def compute_riemann_flux(
        self, UL: np.ndarray, UR: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        return self._riemann.compute_riemann_flux(UL, UR, gas, dx, dt)

    def compute_fluxes(
        self, U: np.ndarray, gas: GasProperties, dx: float, dt: float
    ) -> np.ndarray:
        UL, UR = eno_reconstruct(U, gas.gamma, self._eno_order)
        return self.compute_riemann_flux(UL, UR, gas, dx, dt)


def make_eno_scheme(riemann_name: str, eno_order: int = 2) -> ENOScheme:
    """Factory to create an ENOScheme from a Riemann solver name.

    Parameters
    ----------
    riemann_name : str
        Name of the Riemann flux solver (e.g., "hllc", "roe").
    eno_order : int, default 2
        ENO reconstruction order.

    Returns
    -------
    ENOScheme
        Configured ENO scheme ready for use in the solver.
    """
    from euler1d.schemes.flux import get_flux_solver
    return ENOScheme(get_flux_solver(riemann_name), eno_order)
