"""Numerical flux solvers for the 1D Euler equations.

This package provides eight flux schemes organised into two families:

**Upwind (Riemann-based) schemes** -- These exploit the wave structure of the
Riemann problem to propagate information in the physically correct direction.
Six solvers are available, listed in increasing order of wave resolution:

* :class:`Rusanov` -- single maximum wave speed (most dissipative upwind).
* :class:`HLL` -- two-wave model, single intermediate state.
* :class:`HLLC` -- three-wave model including the contact discontinuity.
* :class:`Roe` -- linearised solver with Harten entropy fix.
* :class:`RoeNoFix` -- Roe without entropy fix (pedagogical comparison).
* :class:`Godunov` -- exact Riemann solver (reference, most expensive).

**Flux vector splitting (FVS) schemes** -- These split the physical flux into
forward and backward parts based on the local Mach number:

* :class:`AUSMPlus` -- Advection Upstream Splitting Method (Liou, 1996).

All composable schemes can be paired with higher-order reconstructions
(MUSCL, ENO, WENO) via the composition principle described in
docs/03_finite_volume.md, §3.5.

**Centred schemes** -- These do not use a Riemann solver.  The flux is a
centred average of the physical fluxes plus an artificial dissipation term:

* :class:`LaxFriedrichs` -- global dissipation proportional to dx/dt.
* :class:`LaxWendroff` -- Richtmyer two-step predictor-corrector (order 2).
* :class:`JST` -- adaptive artificial dissipation with shock sensor (order 2).

These three are **not composable** with higher-order reconstructions.

The helper function :func:`get_flux_solver` instantiates any of the five
composable solvers by name.  The three centred schemes are registered directly
in the top-level scheme registry (``euler1d/schemes/__init__.py``).

See also: docs/04_flux_schemes.md for the full theoretical derivations.
"""

from euler1d.schemes.flux.ausm_plus import AUSMPlus, AUSMPlusUp
from euler1d.schemes.flux.godunov import Godunov
from euler1d.schemes.flux.hll import HLL
from euler1d.schemes.flux.hllc import HLLC
from euler1d.schemes.flux.jst import JST
from euler1d.schemes.flux.lax_friedrichs import LaxFriedrichs
from euler1d.schemes.flux.lax_wendroff import LaxWendroff
from euler1d.schemes.flux.roe import Roe, RoeNoFix
from euler1d.schemes.flux.rusanov import Rusanov

__all__ = [
    "AUSMPlus", "AUSMPlusUp",
    "LaxFriedrichs", "LaxWendroff", "JST", "Rusanov",
    "HLL", "HLLC", "Roe", "RoeNoFix", "Godunov",
    "get_flux_solver",
]

_FLUX_SOLVER_CLASSES = {
    "rusanov": Rusanov,
    "hll": HLL,
    "hllc": HLLC,
    "roe": Roe,
    "roe-nc": RoeNoFix,
    "godunov": Godunov,
    "ausm+": AUSMPlus,
    "ausm+-up": AUSMPlusUp,
}


def get_flux_solver(name: str):
    """Instantiate one of the five composable flux solvers by name.

    The composable solvers are those that can be paired with higher-order
    spatial reconstructions (MUSCL, ENO, WENO): Rusanov, HLL, HLLC, Roe,
    and Godunov.  Centred schemes (Lax-Friedrichs, Lax-Wendroff, JST) are
    not included here; they are registered in the top-level scheme registry.

    Parameters
    ----------
    name : str
        Case-insensitive solver name (e.g. ``"hllc"``, ``"roe"``).

    Returns
    -------
    NumericalScheme
        A fresh instance of the requested flux solver.

    Raises
    ------
    ValueError
        If *name* does not match any known composable solver.
    """
    key = name.lower()
    if key not in _FLUX_SOLVER_CLASSES:
        raise ValueError(
            f"Unknown flux solver {name!r}. "
            f"Available: {list(_FLUX_SOLVER_CLASSES.keys())}"
        )
    return _FLUX_SOLVER_CLASSES[key]()
