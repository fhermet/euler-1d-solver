"""High-order spatial reconstruction methods for finite volume schemes.

This package implements the reconstruction component of the composition principle
described in Chapter 3 (Finite Volume Method): a high-order scheme is obtained by
combining a spatial reconstruction with a Riemann flux solver.

    high-order scheme = reconstruction (this package) + flux solver (schemes.flux)

All reconstruction methods operate on **primitive variables** W = (rho, u, p)
rather than conservative variables U = (rho, rho*u, E). This choice improves
conditioning near discontinuities where conservative variables exhibit coupled
jumps. See docs/05_reconstruction.md, Introduction.

Available reconstructions
-------------------------
- **MUSCL** (order 2): Piecewise-linear with slope limiters (minmod, van-leer,
  superbee, mc, van-albada). See ``muscl.py`` and ``limiters.py``.
- **ENO2** (order 2): Essentially Non-Oscillatory with adaptive stencil
  selection. See ``eno.py``.
- **WENO3** (order 3): Weighted ENO with 2 sub-stencils, JS or Z variants.
  See ``weno3.py``.
- **WENO5** (order 5): Weighted ENO with 3 sub-stencils, JS or Z variants.
  See ``weno5.py``.

Each reconstruction can be composed with any of the five composable Riemann
solvers (Rusanov, HLL, HLLC, Roe, Godunov) via the ``make_*_scheme`` factory
functions or the top-level ``get_scheme()`` registry in ``schemes/__init__.py``.

References
----------
- [van Leer, 1979] MUSCL reconstruction.
- [Harten et al., 1987] ENO methodology.
- [Jiang, Shu, 1996] WENO-JS formulation.
- [Borges et al., 2008] WENO-Z improvement.
"""

from euler1d.schemes.reconstruction.eno import ENOScheme, make_eno_scheme
from euler1d.schemes.reconstruction.limiters import available_limiters, get_limiter
from euler1d.schemes.reconstruction.muscl import MUSCLScheme, make_muscl_scheme
from euler1d.schemes.reconstruction.weno3 import WENO3Scheme, make_weno3_scheme
from euler1d.schemes.reconstruction.weno5 import WENO5Scheme, make_weno5_scheme

__all__ = [
    "MUSCLScheme", "make_muscl_scheme",
    "ENOScheme", "make_eno_scheme",
    "WENO3Scheme", "make_weno3_scheme",
    "WENO5Scheme", "make_weno5_scheme",
    "available_limiters", "get_limiter",
]
