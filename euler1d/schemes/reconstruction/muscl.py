"""MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) reconstruction.

Provides second-order spatial accuracy through piecewise-linear reconstruction
with slope limiting. The reconstruction operates on primitive variables
W = (rho, u, p) and uses a TVD slope limiter to prevent oscillations near
discontinuities while maintaining accuracy in smooth regions.

The composition principle applies: MUSCLScheme pairs this reconstruction with
any composable Riemann flux solver (Rusanov, HLL, HLLC, Roe, Godunov) to
form a complete second-order scheme.

See docs/05_reconstruction.md, section 5.1 for the full derivation, Sweby's
TVD region, and limiter comparison.

References
----------
- [van Leer, 1979] B. van Leer, "Towards the Ultimate Conservative Difference
  Scheme. V. A Second-Order Sequel to Godunov's Method", J. Comput. Phys., 32,
  pp. 101-136.
- [Sweby, 1984] P. K. Sweby, "High Resolution Schemes Using Flux Limiters",
  SIAM J. Numer. Anal., 21(5), pp. 995-1011.
"""

from __future__ import annotations

import numpy as np

from euler1d.config import GasProperties
from euler1d.physics import conservative_to_primitive, primitive_to_conservative
from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.reconstruction.limiters import get_limiter


def muscl_reconstruct(
    U: np.ndarray, gamma: float, limiter_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform MUSCL piecewise-linear reconstruction with slope limiting.

    Converts to primitive variables, computes forward differences and slope
    ratios, applies the limiter function to obtain limited slopes, then
    extrapolates to interface states. Density and pressure are clamped to a
    positivity floor of 1e-10 to prevent non-physical states.

    Parameters
    ----------
    U : ndarray, shape (3, N + 4)
        Conservative variables with 2 ghost cells on each side.
    gamma : float
        Ratio of specific heats.
    limiter_fn : callable
        Slope limiter function phi(r) -> phi_values. Must accept and return
        numpy arrays. See ``limiters.py`` for available limiters.

    Returns
    -------
    UL : ndarray, shape (3, N + 1)
        Reconstructed left states at each of the N+1 interfaces.
    UR : ndarray, shape (3, N + 1)
        Reconstructed right states at each of the N+1 interfaces.

    See Also
    --------
    limiters : Slope limiter functions (minmod, van_leer, superbee, mc).
    MUSCLScheme : Scheme class that pairs this reconstruction with a Riemann solver.
    """
    rho, u, p = conservative_to_primitive(U, gamma)
    W = np.array([rho, u, p])

    dW = np.diff(W, axis=1)

    eps = 1e-30
    r = dW[:, :-1] / (dW[:, 1:] + eps)

    phi = limiter_fn(r)
    sigma = phi * dW[:, 1:]

    W_L = W[:, 1:-2] + 0.5 * sigma[:, :-1]
    W_R = W[:, 2:-1] - 0.5 * sigma[:, 1:]

    W_L[0] = np.maximum(W_L[0], 1e-10)
    W_L[2] = np.maximum(W_L[2], 1e-10)
    W_R[0] = np.maximum(W_R[0], 1e-10)
    W_R[2] = np.maximum(W_R[2], 1e-10)

    UL = primitive_to_conservative(W_L[0], W_L[1], W_L[2], gamma)
    UR = primitive_to_conservative(W_R[0], W_R[1], W_R[2], gamma)

    return UL, UR


class MUSCLScheme(NumericalScheme):
    """Second-order MUSCL scheme combining slope-limited reconstruction with a Riemann solver.

    This class implements the composition principle: the MUSCL reconstruction
    provides piecewise-linear interface states, which are then fed to the
    underlying Riemann solver to compute the numerical flux.

    Properties: order=2, n_ghost=2, default time integrator=RK2.

    Parameters
    ----------
    riemann_solver : NumericalScheme
        The underlying first-order Riemann flux solver (e.g., HLLC, Roe).
    limiter : str, default "van-leer"
        Name of the slope limiter. See ``limiters.available_limiters()``.

    See Also
    --------
    muscl_reconstruct : The reconstruction function used internally.
    make_muscl_scheme : Factory function to create a MUSCLScheme by solver name.
    """

    def __init__(self, riemann_solver: NumericalScheme, limiter: str = "van-leer"):
        self._riemann = riemann_solver
        self._limiter_name = limiter
        self._limiter_fn = get_limiter(limiter)

    @property
    def name(self) -> str:
        return f"MUSCL-{self._riemann.name} ({self._limiter_name})"

    @property
    def order(self) -> int:
        return 2

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
        UL, UR = muscl_reconstruct(U, gas.gamma, self._limiter_fn)
        return self.compute_riemann_flux(UL, UR, gas, dx, dt)


def make_muscl_scheme(riemann_name: str, limiter: str = "van-leer") -> MUSCLScheme:
    """Factory to create a MUSCLScheme from a Riemann solver name.

    Parameters
    ----------
    riemann_name : str
        Name of the Riemann flux solver (e.g., "hllc", "roe").
    limiter : str, default "van-leer"
        Name of the slope limiter.

    Returns
    -------
    MUSCLScheme
        Configured MUSCL scheme ready for use in the solver.
    """
    from euler1d.schemes.flux import get_flux_solver
    return MUSCLScheme(get_flux_solver(riemann_name), limiter=limiter)
