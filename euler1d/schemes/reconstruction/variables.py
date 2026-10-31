"""Variables on which a reconstruction operates, and the shared reconstruction driver.

A reconstruction turns cell averages into left/right states at each interface.
The 1D formulas (limited slope, ENO stencil, WENO weights) are the same
whatever the variables they are applied to; what changes is the set of
variables, component by component:

* **primitive** W = (rho, u, p): the most common choice in practice and the
  default here.  Robust near contacts, but W(U_bar) differs from the cell
  average of W by O(dx^2): beyond order 2 the scheme loses its formal order
  on nonlinear smooth flows (WENO5 drops to order 2).
* **conservative** U = (rho, rho*u, E): exact cell averages, so the formal
  order is kept, but componentwise reconstruction of coupled jumps can
  oscillate near strong discontinuities.
* **characteristic**: at each interface the stencil is projected on the left
  eigenvectors of the Roe matrix, reconstructed wave by wave, and projected
  back.  Keeps the formal order and decouples the waves; the standard choice
  for WENO on strong shocks, at the price of two 3x3 products per interface.

Whatever the variables, a face state with non-positive density or pressure
falls back to the first-order (cell) value.

See docs/05_reconstruction.md, section 5.6.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from euler1d.physics import conservative_to_primitive, primitive_to_conservative

VARIABLES = ("primitive", "conservative", "characteristic")

# kernel(v, dx) -> (v_left, v_right).  v has shape (2*n_ghost, n_var, n_intf):
# v[m] is the cell at offset m in the stencil of each interface, so that
# v[n_ghost - 1] and v[n_ghost] are the cells on each side of the interface.
Kernel = Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]


def check_variables(variables: str) -> str:
    if variables not in VARIABLES:
        raise ValueError(
            f"Unknown reconstruction variables {variables!r}; expected one of {VARIABLES}"
        )
    return variables


def roe_eigenvectors(
    UL: np.ndarray, UR: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Left and right eigenvector matrices of the Roe-averaged Euler Jacobian.

    Returns (L, R), each of shape (3, 3, n), with L = R^-1 at every interface.
    Columns of R are the eigenvectors for u - a, u, u + a.
    """
    rhoL, uL, pL = conservative_to_primitive(UL, gamma)
    rhoR, uR, pR = conservative_to_primitive(UR, gamma)
    sL, sR = np.sqrt(rhoL), np.sqrt(rhoR)
    u = (sL * uL + sR * uR) / (sL + sR)
    H = (sL * (UL[2] + pL) / rhoL + sR * (UR[2] + pR) / rhoR) / (sL + sR)
    a = np.sqrt((gamma - 1.0) * (H - 0.5 * u**2))

    one = np.ones_like(u)
    R = np.array([
        [one, one, one],
        [u - a, u, u + a],
        [H - u * a, 0.5 * u**2, H + u * a],
    ])
    b1 = (gamma - 1.0) / a**2
    b2 = 0.5 * b1 * u**2
    L = 0.5 * np.array([
        [b2 + u / a, -(b1 * u + 1.0 / a), b1],
        [2.0 * (1.0 - b2), 2.0 * b1 * u, -2.0 * b1],
        [b2 - u / a, -(b1 * u - 1.0 / a), b1],
    ])
    return L, R


def _stencils(A: np.ndarray, n_ghost: int) -> np.ndarray:
    """Sliding windows of 2*n_ghost cells, one per interface: (2g, n_var, n_intf)."""
    n_intf = A.shape[1] - 2 * n_ghost + 1
    return np.stack([A[:, m : m + n_intf] for m in range(2 * n_ghost)])


def reconstruct_faces(
    U: np.ndarray,
    gamma: float,
    dx: float,
    kernel: Kernel,
    n_ghost: int,
    variables: str = "primitive",
) -> tuple[np.ndarray, np.ndarray]:
    """Left and right conservative states at the N+1 interfaces.

    Parameters
    ----------
    U : ndarray, shape (3, N + 2*n_ghost)
        Cell averages with ghost cells.
    kernel : callable
        1D reconstruction formula applied componentwise.
    variables : str
        "primitive", "conservative" or "characteristic".
    """
    g = n_ghost
    U_cells = _stencils(U, g)
    U_left_cell, U_right_cell = U_cells[g - 1], U_cells[g]

    if variables == "primitive":
        W = np.array(conservative_to_primitive(U, gamma))
        WL, WR = kernel(_stencils(W, g), dx)
        with np.errstate(invalid="ignore", divide="ignore"):
            UL = primitive_to_conservative(WL[0], WL[1], WL[2], gamma)
            UR = primitive_to_conservative(WR[0], WR[1], WR[2], gamma)
    elif variables == "conservative":
        UL, UR = kernel(U_cells, dx)
    else:
        L, R = roe_eigenvectors(U_left_cell, U_right_cell, gamma)
        C = np.einsum("ijk,mjk->mik", L, U_cells)
        CL, CR = kernel(C, dx)
        UL = np.einsum("ijk,jk->ik", R, CL)
        UR = np.einsum("ijk,jk->ik", R, CR)

    return _first_order_if_unphysical(UL, U_left_cell, gamma), \
        _first_order_if_unphysical(UR, U_right_cell, gamma)


def _first_order_if_unphysical(
    U_face: np.ndarray, U_cell: np.ndarray, gamma: float
) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        rho, _, p = conservative_to_primitive(U_face, gamma)
        bad = ~((rho > 0.0) & (p > 0.0))
    return np.where(bad, U_cell, U_face)
