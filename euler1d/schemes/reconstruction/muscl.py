"""MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) reconstruction.

Provides second-order spatial accuracy through piecewise-linear reconstruction
with slope limiting. The reconstruction operates by default on primitive
variables W = (rho, u, p) (see ``variables.py`` for the other choices) and
uses a TVD slope limiter to prevent oscillations near
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

from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.reconstruction.base import ReconstructedScheme
from euler1d.schemes.reconstruction.limiters import LimiterFn, get_limiter
from euler1d.schemes.reconstruction.variables import reconstruct_faces


def muscl_kernel(v: np.ndarray, limiter_fn: LimiterFn) -> tuple[np.ndarray, np.ndarray]:
    """Limited linear reconstruction on stencils v = (W_{j-1}, W_j, W_{j+1}, W_{j+2}).

    The slope of cell i is phi(r_i) * (W_{i+1} - W_i) with
    r_i = (W_i - W_{i-1}) / (W_{i+1} - W_i).  The left state of interface
    j+1/2 comes from cell j, the right state from cell j+1.
    """
    eps = 1e-30
    d0, d1, d2 = v[1] - v[0], v[2] - v[1], v[3] - v[2]
    slope_j = limiter_fn(d0 / (d1 + eps)) * d1
    slope_jp1 = limiter_fn(d1 / (d2 + eps)) * d2
    return v[1] + 0.5 * slope_j, v[2] - 0.5 * slope_jp1


def muscl_reconstruct(
    U: np.ndarray, gamma: float, limiter_fn: LimiterFn,
) -> tuple[np.ndarray, np.ndarray]:
    """MUSCL reconstruction on primitive variables.

    Parameters
    ----------
    U : ndarray, shape (3, N + 4)
        Conservative variables with 2 ghost cells on each side.
    gamma : float
        Ratio of specific heats.
    limiter_fn : callable
        Slope limiter phi(r).  See ``limiters.py``.

    Returns
    -------
    UL, UR : ndarray, shape (3, N + 1)
        Left and right conservative states at each interface.
    """
    return reconstruct_faces(
        U, gamma, 1.0, lambda v, dx: muscl_kernel(v, limiter_fn), 2, "primitive"
    )


class MUSCLScheme(ReconstructedScheme):
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
    variables : str, default "primitive"
        Reconstructed variables: "primitive", "conservative" or
        "characteristic".

    See Also
    --------
    muscl_kernel : The 1D reconstruction formula.
    make_muscl_scheme : Factory function to create a MUSCLScheme by solver name.
    """

    def __init__(
        self,
        riemann_solver: NumericalScheme,
        limiter: str = "van-leer",
        variables: str = "primitive",
    ):
        super().__init__(riemann_solver, variables)
        self._limiter_name = limiter
        self._limiter_fn = get_limiter(limiter)

    @property
    def label(self) -> str:
        return "MUSCL"

    @property
    def name(self) -> str:
        suffix = "" if self._variables == "primitive" else f" [{self._variables}]"
        return f"MUSCL-{self._riemann.name} ({self._limiter_name}){suffix}"

    @property
    def order(self) -> int:
        return 2

    @property
    def n_ghost(self) -> int:
        return 2

    def kernel(self, v: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
        return muscl_kernel(v, self._limiter_fn)


def make_muscl_scheme(
    riemann_name: str, limiter: str = "van-leer", variables: str = "primitive"
) -> MUSCLScheme:
    """Factory to create a MUSCLScheme from a Riemann solver name.

    Parameters
    ----------
    riemann_name : str
        Name of the Riemann flux solver (e.g., "hllc", "roe").
    limiter : str, default "van-leer"
        Name of the slope limiter.
    variables : str, default "primitive"
        Reconstructed variables.

    Returns
    -------
    MUSCLScheme
        Configured MUSCL scheme ready for use in the solver.
    """
    from euler1d.schemes.flux import get_flux_solver
    return MUSCLScheme(get_flux_solver(riemann_name), limiter=limiter, variables=variables)
