"""ENO (Essentially Non-Oscillatory) reconstruction scheme, order 2.

Implements the ENO methodology of adaptive stencil selection: for each cell,
the linear stencil is extended towards the neighbour with the smallest first
divided difference (i.e., the smoothest local interpolation). This avoids
interpolating across discontinuities without requiring a tunable limiter
function.

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

from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.reconstruction.base import ReconstructedScheme
from euler1d.schemes.reconstruction.variables import reconstruct_faces


def eno_kernel(v: np.ndarray, eno_order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """ENO reconstruction on stencils v = (W_{j-1}, W_j, W_{j+1}, W_{j+2}).

    For ENO2, each cell takes the slope of the stencil, {i-1, i} or
    {i, i+1}, whose first difference is smallest in absolute value: the
    linear stencil is extended towards the smoother side.  ENO1 is the
    piecewise-constant fallback.
    """
    if eno_order == 1:
        return v[1], v[2]
    if eno_order != 2:
        raise ValueError(f"ENO order {eno_order} not supported (use 1 or 2)")
    d0, d1, d2 = v[1] - v[0], v[2] - v[1], v[3] - v[2]
    slope_j = np.where(np.abs(d0) <= np.abs(d1), d0, d1)
    slope_jp1 = np.where(np.abs(d1) <= np.abs(d2), d1, d2)
    return v[1] + 0.5 * slope_j, v[2] - 0.5 * slope_jp1


def eno_reconstruct(
    U: np.ndarray, gamma: float, eno_order: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """ENO reconstruction on primitive variables.

    Parameters
    ----------
    U : ndarray, shape (3, N + 4)
        Conservative variables with 2 ghost cells on each side.
    gamma : float
        Ratio of specific heats.
    eno_order : int, default 2
        1 for piecewise-constant, 2 for piecewise-linear with stencil selection.

    Returns
    -------
    UL, UR : ndarray, shape (3, N + 1)
        Left and right conservative states at each interface.
    """
    return reconstruct_faces(
        U, gamma, 1.0, lambda v, dx: eno_kernel(v, eno_order), 2, "primitive"
    )


class ENOScheme(ReconstructedScheme):
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
    variables : str, default "primitive"
        Reconstructed variables: "primitive", "conservative" or
        "characteristic".

    See Also
    --------
    eno_kernel : The 1D reconstruction formula.
    make_eno_scheme : Factory function to create an ENOScheme by solver name.
    """

    def __init__(
        self,
        riemann_solver: NumericalScheme,
        eno_order: int = 2,
        variables: str = "primitive",
    ):
        super().__init__(riemann_solver, variables)
        self._eno_order = eno_order

    @property
    def label(self) -> str:
        return f"ENO{self._eno_order}"

    @property
    def order(self) -> int:
        return self._eno_order

    @property
    def n_ghost(self) -> int:
        return 2

    def kernel(self, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
        return eno_kernel(v, self._eno_order)


def make_eno_scheme(
    riemann_name: str, eno_order: int = 2, variables: str = "primitive"
) -> ENOScheme:
    """Factory to create an ENOScheme from a Riemann solver name.

    Parameters
    ----------
    riemann_name : str
        Name of the Riemann flux solver (e.g., "hllc", "roe").
    eno_order : int, default 2
        ENO reconstruction order.
    variables : str, default "primitive"
        Reconstructed variables.

    Returns
    -------
    ENOScheme
        Configured ENO scheme ready for use in the solver.
    """
    from euler1d.schemes.flux import get_flux_solver
    return ENOScheme(get_flux_solver(riemann_name), eno_order, variables=variables)
