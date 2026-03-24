"""Scheme registry and factory.

This module is the single entry point for obtaining any numerical scheme by
name.  It implements a **registry pattern**: all available schemes (both
first-order flux solvers and higher-order reconstruction+flux combinations)
are catalogued in internal dictionaries and exposed through the
:func:`get_scheme` factory function.

Naming convention (see docs/03_finite_volume.md, §3.7):

* First-order flux solvers are registered under their bare name:
  ``"rusanov"``, ``"hll"``, ``"hllc"``, ``"roe"``, ``"roe-nc"``,
  ``"godunov"``, ``"lax-friedrichs"``, ``"lax-wendroff"``, ``"jst"``.
  ``"roe-nc"`` is the Roe solver without entropy correction.
* Higher-order schemes follow the pattern ``"reconstruction-flux"``:
  ``"muscl-hllc"``, ``"eno2-roe"``, ``"weno3-hllc"``, ``"weno5-godunov"``,
  ``"wenoz3-hllc"``, ``"wenoz5-roe"``, etc.
* MUSCL schemes accept an optional ``limiter`` keyword argument (default
  ``"van-leer"``); other reconstructions have no extra parameters.

The seven composable flux solvers (Rusanov, HLL, HLLC, Roe, Roe-NC, Godunov,
AUSM+) can be paired with any of the six reconstructions (MUSCL, ENO2, WENO3,
WENOZ3, WENO5, WENOZ5), yielding 42 high-order combinations.  Together with
the 10 standalone schemes, the registry offers 52 schemes in total.

Usage::

    from euler1d.schemes import get_scheme, available_schemes

    scheme = get_scheme("muscl-hllc", limiter="minmod")
    all_names = available_schemes()          # list of all 38 names
    order2 = available_schemes(order=2)      # only order-2 schemes

See also: docs/03_finite_volume.md, §3.7 (scheme registry).
"""

from __future__ import annotations

from euler1d.schemes.base import NumericalScheme
from euler1d.schemes.flux import (
    AUSMPlus, AUSMPlusUp, Godunov, HLL, HLLC, JST, LaxFriedrichs, LaxWendroff, Roe, RoeNoFix, Rusanov,
)
from euler1d.schemes.reconstruction import (
    available_limiters,
    make_eno_scheme,
    make_muscl_scheme,
    make_weno3_scheme,
    make_weno5_scheme,
)

# --- Flux solvers (order 1, instantiated directly) ---

SCHEME_REGISTRY: dict[str, type[NumericalScheme]] = {
    "lax-friedrichs": LaxFriedrichs,
    "rusanov": Rusanov,
    "hll": HLL,
    "hllc": HLLC,
    "roe": Roe,
    "roe-nc": RoeNoFix,
    "godunov": Godunov,
    "ausm+": AUSMPlus,
    "ausm+-up": AUSMPlusUp,
    "lax-wendroff": LaxWendroff,
    "jst": JST,
}

# --- Factory-based schemes (reconstruction + flux) ---

_MUSCL_ENTRIES: dict[str, str] = {
    "muscl-rusanov": "rusanov",
    "muscl-hll": "hll",
    "muscl-hllc": "hllc",
    "muscl-roe": "roe",
    "muscl-roe-nc": "roe-nc",
    "muscl-godunov": "godunov",
    "muscl-ausm+": "ausm+",
    "muscl-ausm+-up": "ausm+-up",
}

_ENO_ENTRIES: dict[str, tuple[str, int]] = {
    "eno2-rusanov": ("rusanov", 2),
    "eno2-hll": ("hll", 2),
    "eno2-hllc": ("hllc", 2),
    "eno2-roe": ("roe", 2),
    "eno2-roe-nc": ("roe-nc", 2),
    "eno2-godunov": ("godunov", 2),
    "eno2-ausm+": ("ausm+", 2),
    "eno2-ausm+-up": ("ausm+-up", 2),
}

_WENO3_ENTRIES: dict[str, tuple[str, str]] = {
    "weno3-rusanov": ("rusanov", "JS"),
    "weno3-hll": ("hll", "JS"),
    "weno3-hllc": ("hllc", "JS"),
    "weno3-roe": ("roe", "JS"),
    "weno3-roe-nc": ("roe-nc", "JS"),
    "weno3-godunov": ("godunov", "JS"),
    "weno3-ausm+": ("ausm+", "JS"),
    "weno3-ausm+-up": ("ausm+-up", "JS"),
    "wenoz3-rusanov": ("rusanov", "Z"),
    "wenoz3-hll": ("hll", "Z"),
    "wenoz3-hllc": ("hllc", "Z"),
    "wenoz3-roe": ("roe", "Z"),
    "wenoz3-roe-nc": ("roe-nc", "Z"),
    "wenoz3-godunov": ("godunov", "Z"),
    "wenoz3-ausm+": ("ausm+", "Z"),
    "wenoz3-ausm+-up": ("ausm+-up", "Z"),
}

_WENO5_ENTRIES: dict[str, tuple[str, str]] = {
    "weno5-rusanov": ("rusanov", "JS"),
    "weno5-hll": ("hll", "JS"),
    "weno5-hllc": ("hllc", "JS"),
    "weno5-roe": ("roe", "JS"),
    "weno5-roe-nc": ("roe-nc", "JS"),
    "weno5-godunov": ("godunov", "JS"),
    "weno5-ausm+": ("ausm+", "JS"),
    "weno5-ausm+-up": ("ausm+-up", "JS"),
    "wenoz5-rusanov": ("rusanov", "Z"),
    "wenoz5-hll": ("hll", "Z"),
    "wenoz5-hllc": ("hllc", "Z"),
    "wenoz5-roe": ("roe", "Z"),
    "wenoz5-roe-nc": ("roe-nc", "Z"),
    "wenoz5-godunov": ("godunov", "Z"),
    "wenoz5-ausm+": ("ausm+", "Z"),
    "wenoz5-ausm+-up": ("ausm+-up", "Z"),
}


def get_scheme(name: str, **kwargs) -> NumericalScheme:
    """Instantiate a scheme by name.

    For MUSCL schemes, pass limiter="minmod" etc. via kwargs.
    """
    key = name.lower()
    if key in _MUSCL_ENTRIES:
        riemann = _MUSCL_ENTRIES[key]
        return make_muscl_scheme(riemann, **kwargs)
    if key in _ENO_ENTRIES:
        riemann, eno_order = _ENO_ENTRIES[key]
        return make_eno_scheme(riemann, eno_order)
    if key in _WENO3_ENTRIES:
        riemann, variant = _WENO3_ENTRIES[key]
        return make_weno3_scheme(riemann, variant=variant)
    if key in _WENO5_ENTRIES:
        riemann, variant = _WENO5_ENTRIES[key]
        return make_weno5_scheme(riemann, variant=variant)
    if key not in SCHEME_REGISTRY:
        raise ValueError(
            f"Unknown scheme {name!r}. Available: {available_schemes()}"
        )
    return SCHEME_REGISTRY[key](**kwargs)


def available_schemes(order: int | None = None) -> list[str]:
    """Return list of registered scheme names, optionally filtered by order."""
    all_keys = (
        list(SCHEME_REGISTRY.keys())
        + list(_MUSCL_ENTRIES.keys())
        + list(_ENO_ENTRIES.keys())
        + list(_WENO3_ENTRIES.keys())
        + list(_WENO5_ENTRIES.keys())
    )
    if order is None:
        return all_keys
    result = []
    for name in all_keys:
        s = get_scheme(name)
        if s.order == order:
            result.append(name)
    return result


def scheme_label(key: str) -> str:
    """Return a rich display label for a scheme registry key."""
    s = get_scheme(key)
    if key.startswith("muscl"):
        base_name = s.name.split(" (")[0]
    else:
        base_name = s.name
    return f"{base_name}  ({s.scheme_type}, ordre {s.order})"
