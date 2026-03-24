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

The reconstruction operates on primitive variables W = (rho, u, p).

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

from euler1d.config import GasProperties
from euler1d.physics import conservative_to_primitive, primitive_to_conservative
from euler1d.schemes.base import NumericalScheme


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


def weno3_reconstruct(
    U: np.ndarray, gamma: float, variant: str = "JS", dx: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Perform WENO3 reconstruction on primitive variables using 2 sub-stencils.

    Blends two candidate stencils S0={i-1,i} and S1={i,i+1} with nonlinear
    weights derived from smoothness indicators (squared first differences).
    In smooth regions, the weights converge to the ideal values (1/3, 2/3 for
    left; 2/3, 1/3 for right), achieving third-order accuracy. Near
    discontinuities, the contaminated stencil is effectively excluded.

    Density and pressure are clamped to 1e-10 for positivity.

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
    UL : ndarray, shape (3, N + 1)
        Reconstructed left states at each interface.
    UR : ndarray, shape (3, N + 1)
        Reconstructed right states at each interface.

    See Also
    --------
    WENO3Scheme : Scheme class that pairs this reconstruction with a Riemann solver.
    weno5_reconstruct : Higher-order (5th) WENO reconstruction.
    """
    rho, u, p = conservative_to_primitive(U, gamma)
    W = np.array([rho, u, p])

    D = np.diff(W, axis=1)
    n_intf = W.shape[1] - 3

    if variant == "Z":
        eps_z = dx**2
        def weight_fn(b0, b1, d0, d1):
            return _weno_weights_z(b0, b1, d0, d1, eps=eps_z)
    else:
        weight_fn = _weno_weights_js

    D0_L = D[:, 0:n_intf]
    D1_L = D[:, 1:n_intf+1]
    w0_L, w1_L = weight_fn(D0_L**2, D1_L**2, 1.0/3.0, 2.0/3.0)
    W_L = W[:, 1:n_intf+1] + 0.5 * (w0_L * D0_L + w1_L * D1_L)

    D0_R = D[:, 1:n_intf+1]
    D1_R = D[:, 2:n_intf+2]
    w0_R, w1_R = weight_fn(D0_R**2, D1_R**2, 2.0/3.0, 1.0/3.0)
    W_R = W[:, 2:n_intf+2] - 0.5 * (w0_R * D0_R + w1_R * D1_R)

    W_L[0] = np.maximum(W_L[0], 1e-10)
    W_L[2] = np.maximum(W_L[2], 1e-10)
    W_R[0] = np.maximum(W_R[0], 1e-10)
    W_R[2] = np.maximum(W_R[2], 1e-10)

    UL = primitive_to_conservative(W_L[0], W_L[1], W_L[2], gamma)
    UR = primitive_to_conservative(W_R[0], W_R[1], W_R[2], gamma)

    return UL, UR


class WENO3Scheme(NumericalScheme):
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

    See Also
    --------
    weno3_reconstruct : The reconstruction function used internally.
    make_weno3_scheme : Factory function to create a WENO3Scheme by solver name.
    """

    def __init__(self, riemann_solver: NumericalScheme, variant: str = "JS"):
        self._riemann = riemann_solver
        self._variant = variant

    @property
    def name(self) -> str:
        tag = "WENO3" if self._variant == "JS" else "WENO3-Z"
        return f"{tag}-{self._riemann.name}"

    @property
    def order(self) -> int:
        return 3

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
        UL, UR = weno3_reconstruct(U, gas.gamma, variant=self._variant, dx=dx)
        return self.compute_riemann_flux(UL, UR, gas, dx, dt)


def make_weno3_scheme(riemann_name: str, variant: str = "JS") -> WENO3Scheme:
    """Factory to create a WENO3Scheme from a Riemann solver name.

    Parameters
    ----------
    riemann_name : str
        Name of the Riemann flux solver (e.g., "hllc", "roe").
    variant : str, default "JS"
        Weight variant: "JS" for Jiang-Shu, "Z" for WENO-Z.

    Returns
    -------
    WENO3Scheme
        Configured WENO3 scheme ready for use in the solver.
    """
    from euler1d.schemes.flux import get_flux_solver
    return WENO3Scheme(get_flux_solver(riemann_name), variant=variant)
