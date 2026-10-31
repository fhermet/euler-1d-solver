"""WENO3 (Weighted ENO, order 3) reconstruction scheme with 2 sub-stencils.

Implements the WENO methodology: instead of selecting a single stencil (ENO),
all candidate stencils are blended with nonlinear weights that adapt to the
local smoothness. In smooth regions, the weights converge to their optimal
(linear) values, yielding third-order accuracy. Near discontinuities, the
weight of the stencil crossing the discontinuity tends to zero, recovering
ENO-like behavior.

Two weight variants are available:
- **JS** (Jiang-Shu): classical weights with a fixed regularization parameter.
  Loses accuracy at critical points where derivatives vanish.
- **Z** (WENO-Z, Borges et al.): uses a global smoothness indicator tau to
  restore optimal order at critical points, with eps = dx^2.

The reconstruction operates by default on primitive variables W = (rho, u, p)
(see ``variables.py`` for the other choices).

See docs/05_reconstruction.md, section 5.3 for full derivation and analysis.

References
----------
- [Jiang, Shu, 1996] G.-S. Jiang, C.-W. Shu, "Efficient Implementation of
  Weighted ENO Schemes", J. Comput. Phys., 126, pp. 202-228.
- [Borges et al., 2008] R. Borges et al., "An Improved Weighted Essentially
  Non-Oscillatory Scheme for Hyperbolic Conservation Laws", J. Comput. Phys.,
  227, pp. 3191-3211.
"""

from __future__ import annotations

import numpy as np

from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.reconstruction.base import ReconstructedScheme
from euler1d.schemes.reconstruction.variables import reconstruct_faces


def _weno_weights_js(
    beta0: np.ndarray, beta1: np.ndarray, d0: float, d1: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jiang-Shu (WENO-JS) nonlinear weights for 2 sub-stencils.

    Uses the classical formulation with a fixed regularization parameter
    eps = 1e-6. May lose accuracy at critical points where smoothness
    indicators are both small.

    Parameters
    ----------
    beta0, beta1 : ndarray
        Smoothness indicators for each sub-stencil.
    d0, d1 : float
        Ideal (linear) weights.

    Returns
    -------
    w0, w1 : ndarray
        Normalized nonlinear weights.

    Reference: [Jiang, Shu, 1996].
    """
    eps = 1e-6
    alpha0 = d0 / (eps + beta0)**2
    alpha1 = d1 / (eps + beta1)**2
    alpha_sum = alpha0 + alpha1
    return alpha0 / alpha_sum, alpha1 / alpha_sum


def _weno_weights_z(
    beta0: np.ndarray, beta1: np.ndarray, d0: float, d1: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Borges (WENO-Z) nonlinear weights for 2 sub-stencils.

    Uses a global smoothness indicator tau = |beta1 - beta0| to restore
    optimal order at critical points. The regularization parameter eps should
    be set to dx^2 for mesh-adaptive behavior.

    Parameters
    ----------
    beta0, beta1 : ndarray
        Smoothness indicators for each sub-stencil.
    d0, d1 : float
        Ideal (linear) weights.
    eps : float, default 1e-6
        Regularization parameter (should be dx^2 for WENO-Z).

    Returns
    -------
    w0, w1 : ndarray
        Normalized nonlinear weights.

    Reference: [Borges et al., 2008].
    """
    tau = np.abs(beta1 - beta0)
    alpha0 = d0 * (1.0 + (tau / (beta0 + eps))**2)
    alpha1 = d1 * (1.0 + (tau / (beta1 + eps))**2)
    alpha_sum = alpha0 + alpha1
    return alpha0 / alpha_sum, alpha1 / alpha_sum


WENO_VARIANTS = ("JS", "Z")


def weno3_kernel(
    v: np.ndarray, dx: float, variant: str = "JS"
) -> tuple[np.ndarray, np.ndarray]:
    """WENO3 reconstruction on stencils v = (W_{j-1}, W_j, W_{j+1}, W_{j+2}).

    Blends the two linear candidates S0 = {i-1, i} and S1 = {i, i+1} with
    nonlinear weights built on the squared first differences.  Ideal weights
    are (1/3, 2/3) for the left state of j+1/2 (from cell j) and (2/3, 1/3)
    for the right state (from cell j+1).
    """
    if variant == "Z":
        eps_z = dx**2

        def weight_fn(b0, b1, d0, d1):
            return _weno_weights_z(b0, b1, d0, d1, eps=eps_z)
    else:
        weight_fn = _weno_weights_js

    D0, D1, D2 = v[1] - v[0], v[2] - v[1], v[3] - v[2]
    w0_L, w1_L = weight_fn(D0**2, D1**2, 1.0 / 3.0, 2.0 / 3.0)
    w0_R, w1_R = weight_fn(D1**2, D2**2, 2.0 / 3.0, 1.0 / 3.0)
    W_L = v[1] + 0.5 * (w0_L * D0 + w1_L * D1)
    W_R = v[2] - 0.5 * (w0_R * D1 + w1_R * D2)
    return W_L, W_R


def weno3_reconstruct(
    U: np.ndarray, gamma: float, variant: str = "JS", dx: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """WENO3 reconstruction on primitive variables.

    Parameters
    ----------
    U : ndarray, shape (3, N + 4)
        Conservative variables with 2 ghost cells on each side.
    gamma : float
        Ratio of specific heats.
    variant : str, default "JS"
        Weight variant: "JS" for Jiang-Shu, "Z" for WENO-Z (Borges et al.).
    dx : float, default 1.0
        Cell size, used by WENO-Z variant for eps = dx^2.

    Returns
    -------
    UL, UR : ndarray, shape (3, N + 1)
        Left and right conservative states at each interface.
    """
    return reconstruct_faces(
        U, gamma, dx, lambda v, h: weno3_kernel(v, h, variant), 2, "primitive"
    )


class WENO3Scheme(ReconstructedScheme):
    """Third-order WENO3 scheme combining weighted reconstruction with a Riemann solver.

    Applies WENO3 reconstruction to compute interface states, then delegates
    flux computation to the underlying Riemann solver. Supports both JS and Z
    weight variants.

    Properties: order=3, n_ghost=2, default time integrator=SSP-RK3.

    Parameters
    ----------
    riemann_solver : NumericalScheme
        The underlying first-order Riemann flux solver.
    variant : str, default "JS"
        Weight variant: "JS" for Jiang-Shu, "Z" for WENO-Z.
    variables : str, default "primitive"
        Reconstructed variables: "primitive", "conservative" or
        "characteristic".

    See Also
    --------
    weno3_kernel : The 1D reconstruction formula.
    make_weno3_scheme : Factory function to create a WENO3Scheme by solver name.
    """

    def __init__(
        self,
        riemann_solver: NumericalScheme,
        variant: str = "JS",
        variables: str = "primitive",
    ):
        super().__init__(riemann_solver, variables)
        self._variant = variant

    @property
    def label(self) -> str:
        return "WENO3" if self._variant == "JS" else "WENO3-Z"

    @property
    def order(self) -> int:
        return 3

    @property
    def n_ghost(self) -> int:
        return 2

    def kernel(self, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
        return weno3_kernel(v, dx, self._variant)


def make_weno3_scheme(
    riemann_name: str, variant: str = "JS", variables: str = "primitive"
) -> WENO3Scheme:
    """Factory to create a WENO3Scheme from a Riemann solver name.

    Parameters
    ----------
    riemann_name : str
        Name of the Riemann flux solver (e.g., "hllc", "roe").
    variant : str, default "JS"
        Weight variant: "JS" for Jiang-Shu, "Z" for WENO-Z.
    variables : str, default "primitive"
        Reconstructed variables.

    Returns
    -------
    WENO3Scheme
        Configured WENO3 scheme ready for use in the solver.
    """
    from euler1d.schemes.flux import get_flux_solver
    return WENO3Scheme(get_flux_solver(riemann_name), variant=variant, variables=variables)
