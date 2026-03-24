"""Boundary conditions via ghost cells.

Finite volume schemes require values beyond the physical domain boundaries
to evaluate fluxes at boundary interfaces. Ghost cells are fictitious cells
appended on each side of the domain and filled according to the chosen
boundary condition. The extended array has shape (3, N + 2*n_ghost), where
n_ghost depends on the stencil width of the spatial scheme (1 for first-order,
2 for MUSCL/ENO/WENO3, 3 for WENO5).

Three boundary condition types are supported:

* **transmissive** (zero-gradient) -- ghost cells copy the nearest interior
  cell.  Physically this models an open boundary: waves leave the domain
  freely without reflection.  Used for Riemann-type problems (Sod, Lax, etc.).

* **reflective** -- ghost cells mirror the interior with reversed momentum.
  This enforces zero normal velocity at the boundary, modelling a rigid wall.
  Waves hitting the boundary are reflected back into the domain.

* **periodic** -- ghost cells on the left copy interior cells from the right
  side and vice-versa.  The domain wraps around, which is appropriate for
  smooth periodic test cases (entropy wave, acoustic wave).

See also: docs/03_finite_volume.md, section 3.6.
"""

from __future__ import annotations

import numpy as np


def apply_boundary_conditions(
    U: np.ndarray, bc_type: str, n_ghost: int = 1
) -> np.ndarray:
    """Extend the interior solution with ghost cells for boundary treatment.

    Takes the conservative variable array on the N physical cells and returns
    an extended array of shape (3, N + 2*n_ghost) with ghost cells filled
    according to the requested boundary condition.  The interior data occupies
    columns ``n_ghost`` to ``n_ghost + N - 1`` of the returned array.

    The number of ghost cells is dictated by the spatial scheme's stencil
    width (see ``NumericalScheme.n_ghost``).

    Boundary condition types
    ------------------------
    * ``"transmissive"`` -- Zero-gradient extrapolation.  Each ghost cell is
      a copy of the nearest interior cell.  Models an open (non-reflecting)
      boundary.
    * ``"reflective"`` -- Mirror symmetry with momentum sign reversal.
      Enforces zero normal velocity at the wall, reflecting incoming waves.
    * ``"periodic"`` -- Left ghosts copy from the right interior cells and
      right ghosts copy from the left interior cells, closing the domain
      into a loop.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative variables (rho, rho*u, E) on interior cells.
    bc_type : str
        One of ``"transmissive"``, ``"reflective"``, or ``"periodic"``.
    n_ghost : int, optional
        Number of ghost cells on each side (default 1).

    Returns
    -------
    U_ext : ndarray, shape (3, N + 2*n_ghost)
        Extended array with ghost cells populated.

    Raises
    ------
    ValueError
        If *bc_type* is not one of the recognised strings.

    See Also
    --------
    euler1d.schemes.base.NumericalScheme.n_ghost :
        Property that specifies how many ghost cells a scheme requires.
    docs/03_finite_volume.md, §3.6 :
        Mathematical description of each boundary condition.
    """
    n_vars, n_cells = U.shape
    U_ext = np.empty((n_vars, n_cells + 2 * n_ghost))
    U_ext[:, n_ghost : n_ghost + n_cells] = U

    if bc_type == "transmissive":
        for i in range(n_ghost):
            U_ext[:, i] = U[:, 0]
            U_ext[:, n_ghost + n_cells + i] = U[:, -1]
    elif bc_type == "reflective":
        for i in range(n_ghost):
            U_ext[:, n_ghost - 1 - i] = U[:, i]
            U_ext[1, n_ghost - 1 - i] *= -1.0  # reverse momentum
            U_ext[:, n_ghost + n_cells + i] = U[:, n_cells - 1 - i]
            U_ext[1, n_ghost + n_cells + i] *= -1.0
    elif bc_type == "periodic":
        for i in range(n_ghost):
            U_ext[:, n_ghost - 1 - i] = U[:, n_cells - 1 - i]
            U_ext[:, n_ghost + n_cells + i] = U[:, i]
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type!r}")

    return U_ext
