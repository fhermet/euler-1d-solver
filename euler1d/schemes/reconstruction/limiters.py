"""Slope limiter functions for MUSCL piecewise-linear reconstruction.

A slope limiter phi(r) controls the gradient in each cell to enforce the TVD
(Total Variation Diminishing) property, preventing spurious oscillations near
discontinuities while preserving second-order accuracy in smooth regions.

The argument *r* is the ratio of consecutive forward differences:
r_i = Delta W_{i-1/2} / Delta W_{i+1/2}. The limiter returns a value
phi(r) >= 0 that scales the slope. For TVD stability, phi(r) must lie within
Sweby's TVD region, bounded by minmod (lower) and superbee (upper).

All limiters are vectorized (accept and return numpy arrays) and operate
component-wise on the three primitive variables (rho, u, p).

See docs/05_reconstruction.md, section 5.1 for the full mathematical treatment,
Sweby's TVD region, and a comparative table.

References
----------
- [Sweby, 1984] P. K. Sweby, "High Resolution Schemes Using Flux Limiters",
  SIAM J. Numer. Anal., 21(5), pp. 995-1011.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

LimiterFn = Callable[[np.ndarray], np.ndarray]


def minmod(r: np.ndarray) -> np.ndarray:
    """Minmod limiter: the most dissipative TVD limiter (lower bound of Sweby region).

    Selects the smallest slope in absolute value, clipping to zero at extrema.
    Very robust but overly diffusive on smooth gradients and contact waves.

    Reference: [Sweby, 1984], see docs/05_reconstruction.md, section 5.1.
    """
    return np.maximum(0.0, np.minimum(1.0, r))


def van_leer(r: np.ndarray) -> np.ndarray:
    """Van Leer limiter: smooth (differentiable) limiter in the center of Sweby region.

    The only common limiter that is continuously differentiable, which benefits
    convergence of implicit time integrators. Provides a good compromise between
    dissipation and accuracy for most practical applications.

    Reference: [van Leer, 1979], see docs/05_reconstruction.md, section 5.1.
    """
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def superbee(r: np.ndarray) -> np.ndarray:
    """Superbee limiter: the least dissipative TVD limiter (upper bound of Sweby region).

    Maximizes the slope within TVD constraints, yielding the sharpest resolution
    of discontinuities. However, it can artificially steepen smooth profiles
    (compressive/staircase effect), which may be undesirable for smooth solutions.

    Reference: [Roe, 1986], see docs/05_reconstruction.md, section 5.1.
    """
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))


def mc(r: np.ndarray) -> np.ndarray:
    """MC (Monotonized Central) limiter: centered slope with TVD constraints.

    Uses the centered-difference slope (1+r)/2 but clips it to the TVD bounds
    2r and 2. Sits between van Leer and superbee in Sweby's region, offering
    better accuracy than van Leer in smooth regions without the compressive
    artifacts of superbee.

    Reference: [van Leer, 1977], see docs/05_reconstruction.md, section 5.1.
    """
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))


def van_albada(r: np.ndarray) -> np.ndarray:
    """Van Albada limiter: smooth symmetric limiter with quadratic blending.

    Uses a rational function phi(r) = (r^2 + r) / (r^2 + 1) that is
    continuously differentiable like van Leer but with a slightly more
    dissipative profile. The quadratic terms ensure smooth transition through
    r = 1 (uniform gradient). Returns 0 for r <= 0 (local extrema).

    Compared to van Leer, van Albada is slightly more diffusive near
    discontinuities but produces smoother solutions. It sits in the lower
    part of Sweby's TVD region, between minmod and van Leer.

    Reference: [van Albada et al., 1982] G. D. van Albada, B. van Leer,
    W. W. Roberts, "A comparative study of computational methods in cosmic
    gas dynamics", Astronomy & Astrophysics, 108, pp. 76-84.
    """
    return np.where(r > 0.0, (r**2 + r) / (r**2 + 1.0), 0.0)


LIMITER_REGISTRY: dict[str, LimiterFn] = {
    "minmod": minmod,
    "van-leer": van_leer,
    "superbee": superbee,
    "mc": mc,
    "van-albada": van_albada,
}


def get_limiter(name: str) -> LimiterFn:
    """Retrieve a slope limiter function by name.

    Parameters
    ----------
    name : str
        Case-insensitive limiter name. One of: "minmod", "van-leer",
        "superbee", "mc", "van-albada".

    Returns
    -------
    LimiterFn
        Vectorized limiter function phi(r) -> phi_values.

    Raises
    ------
    ValueError
        If the name does not match any registered limiter.
    """
    key = name.lower()
    if key not in LIMITER_REGISTRY:
        raise ValueError(
            f"Unknown limiter {name!r}. Available: {available_limiters()}"
        )
    return LIMITER_REGISTRY[key]


def available_limiters() -> list[str]:
    """Return the list of registered limiter names."""
    return list(LIMITER_REGISTRY.keys())
