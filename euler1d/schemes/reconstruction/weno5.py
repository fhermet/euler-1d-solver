"""WENO5 (Weighted ENO, order 5) reconstruction scheme with 3 sub-stencils.

Implements fifth-order WENO reconstruction using three 3-cell sub-stencils.
Each sub-stencil provides a quadratic polynomial interpolation; the final
reconstruction is a nonlinear combination whose weights adapt to the local
smoothness. In smooth regions, the scheme achieves fifth-order accuracy.
Near discontinuities, stencils crossing the discontinuity are effectively
excluded via near-zero weights.

The smoothness indicators include both first- and second-derivative terms,
making them more sensitive to local regularity than the WENO3 indicators.

Two weight variants are available:
- **JS** (Jiang-Shu): classical weights with eps = 1e-6. Loses accuracy at
  critical points.
- **Z** (WENO-Z, Borges et al.): uses tau_5 = |beta_0 - beta_2| to restore
  optimal order at critical points, with eps = dx^2.

The reconstruction operates by default on primitive variables W = (rho, u, p),
which limits it to order 2 on nonlinear flows; conservative or characteristic
variables keep order 5 (see ``variables.py``).  It requires 3 ghost cells on
each side.

See docs/05_reconstruction.md, section 5.4 for full derivation, polynomial
formulas, and JS vs Z comparison.

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


def _weno5_weights(
    beta0: np.ndarray, beta1: np.ndarray, beta2: np.ndarray,
    d0: float, d1: float, d2: float,
    eps: float, variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute WENO5 nonlinear weights for 3 sub-stencils (JS or Z variant).

    For JS: classical formulation alpha_k = d_k / (eps + beta_k)^2.
    For Z: uses global indicator tau_5 = |beta_0 - beta_2| to restore optimal
    convergence at critical points.

    Parameters
    ----------
    beta0, beta1, beta2 : ndarray
        Smoothness indicators for each sub-stencil.
    d0, d1, d2 : float
        Ideal (linear) weights.
    eps : float
        Regularization parameter (1e-6 for JS, dx^2 for Z).
    variant : str
        "JS" for Jiang-Shu, "Z" for WENO-Z.

    Returns
    -------
    w0, w1, w2 : ndarray
        Normalized nonlinear weights.

    References: [Jiang, Shu, 1996], [Borges et al., 2008].
    """
    if variant == "Z":
        tau5 = np.abs(beta0 - beta2)
        alpha0 = d0 * (1.0 + (tau5 / (beta0 + eps))**2)
        alpha1 = d1 * (1.0 + (tau5 / (beta1 + eps))**2)
        alpha2 = d2 * (1.0 + (tau5 / (beta2 + eps))**2)
    else:
        alpha0 = d0 / (eps + beta0)**2
        alpha1 = d1 / (eps + beta1)**2
        alpha2 = d2 / (eps + beta2)**2
    alpha_sum = alpha0 + alpha1 + alpha2
    return alpha0 / alpha_sum, alpha1 / alpha_sum, alpha2 / alpha_sum


def weno5_kernel(
    v: np.ndarray, dx: float, variant: str = "JS"
) -> tuple[np.ndarray, np.ndarray]:
    """WENO5 reconstruction on stencils v = (W_{j-2}, ..., W_{j+3}).

    Blends three quadratic polynomials from sub-stencils S0={i,i+1,i+2},
    S1={i-1,i,i+1}, S2={i-2,i-1,i} with nonlinear weights derived from
    smoothness indicators that include both first- and second-derivative
    terms.  The left state of interface j+1/2 is reconstructed from cell j,
    the right state from cell j+1 with the mirrored formulas.
    """
    eps = dx**2 if variant == "Z" else 1e-6

    # ---- Left state at interface j+1/2, from cell j ----
    jm2, jm1, j0, jp1, jp2 = v[0], v[1], v[2], v[3], v[4]

    beta0_L = (13.0/12.0)*(j0 - 2.0*jp1 + jp2)**2 \
            + 0.25*(3.0*j0 - 4.0*jp1 + jp2)**2
    beta1_L = (13.0/12.0)*(jm1 - 2.0*j0 + jp1)**2 \
            + 0.25*(jm1 - jp1)**2
    beta2_L = (13.0/12.0)*(jm2 - 2.0*jm1 + j0)**2 \
            + 0.25*(jm2 - 4.0*jm1 + 3.0*j0)**2

    p0_L =  1.0/3.0*j0  + 5.0/6.0*jp1 - 1.0/6.0*jp2
    p1_L = -1.0/6.0*jm1 + 5.0/6.0*j0  + 1.0/3.0*jp1
    p2_L =  1.0/3.0*jm2 - 7.0/6.0*jm1 + 11.0/6.0*j0

    w0_L, w1_L, w2_L = _weno5_weights(
        beta0_L, beta1_L, beta2_L, 3.0/10.0, 3.0/5.0, 1.0/10.0, eps, variant
    )
    W_L = w0_L * p0_L + w1_L * p1_L + w2_L * p2_L

    # ---- Right state at interface j+1/2, from cell j+1 ----
    # Stencils (in physical order, leftmost to rightmost):
    #   S2 = {j-1, j, j+1}     -- biased left
    #   S1 = {j, j+1, j+2}     -- central
    #   S0 = {j+1, j+2, j+3}   -- biased right
    #
    # Notation: k0 = W[j+1], km1 = W[j+2], km2 = W[j+3],
    #           kp1 = W[j],  kp2 = W[j-1]
    kp2, kp1, k0, km1, km2 = v[1], v[2], v[3], v[4], v[5]

    beta0_R = (13.0/12.0)*(k0 - 2.0*km1 + km2)**2 \
            + 0.25*(3.0*k0 - 4.0*km1 + km2)**2
    beta1_R = (13.0/12.0)*(kp1 - 2.0*k0 + km1)**2 \
            + 0.25*(kp1 - km1)**2
    beta2_R = (13.0/12.0)*(kp2 - 2.0*kp1 + k0)**2 \
            + 0.25*(kp2 - 4.0*kp1 + 3.0*k0)**2

    p0_R = 11.0/6.0*k0 - 7.0/6.0*km1 + 1.0/3.0*km2
    p1_R =  1.0/3.0*kp1 + 5.0/6.0*k0 - 1.0/6.0*km1
    p2_R = -1.0/6.0*kp2 + 5.0/6.0*kp1 + 1.0/3.0*k0

    w0_R, w1_R, w2_R = _weno5_weights(
        beta0_R, beta1_R, beta2_R, 1.0/10.0, 3.0/5.0, 3.0/10.0, eps, variant
    )
    W_R = w0_R * p0_R + w1_R * p1_R + w2_R * p2_R

    return W_L, W_R


def weno5_reconstruct(
    U: np.ndarray, gamma: float, variant: str = "JS", dx: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """WENO5 reconstruction on primitive variables.

    Parameters
    ----------
    U : ndarray, shape (3, N + 6)
        Conservative variables with 3 ghost cells on each side.
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
        U, gamma, dx, lambda v, h: weno5_kernel(v, h, variant), 3, "primitive"
    )


class WENO5Scheme(ReconstructedScheme):
    """Fifth-order WENO5 scheme combining weighted reconstruction with a Riemann solver.

    Applies WENO5 reconstruction (3 sub-stencils, quadratic polynomials) to
    compute interface states, then delegates flux computation to the underlying
    Riemann solver. Supports both JS and Z weight variants.

    The formal order 5 is reached only with conservative or characteristic
    variables: on primitive variables, W(U_bar) is a second-order
    approximation of the cell average of W.

    Properties: order=5, n_ghost=3, default time integrator=SSP-RK3.

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
    weno5_kernel : The 1D reconstruction formula.
    make_weno5_scheme : Factory function to create a WENO5Scheme by solver name.
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
        return "WENO5" if self._variant == "JS" else "WENO5-Z"

    @property
    def order(self) -> int:
        return 5

    @property
    def n_ghost(self) -> int:
        return 3

    def kernel(self, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
        return weno5_kernel(v, dx, self._variant)


def make_weno5_scheme(
    riemann_name: str, variant: str = "JS", variables: str = "primitive"
) -> WENO5Scheme:
    """Factory to create a WENO5Scheme from a Riemann solver name.

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
    WENO5Scheme
        Configured WENO5 scheme ready for use in the solver.
    """
    from euler1d.schemes.flux import get_flux_solver
    return WENO5Scheme(get_flux_solver(riemann_name), variant=variant, variables=variables)
