"""Figures of the finite-volume walkthrough page."""

import numpy as np
import plotly.graph_objects as go
import pytest

from euler1d.fv_walkthrough import (
    advance, default_interface, exact_waves, flat_cells, flux_breakdown,
    interface_states, model_waves, rk_stages, sod_setup, stage_update, time_step,
)
from euler1d.physics import conservative_to_primitive
from euler1d.schemes import get_scheme
from euler1d.solver import initial_state
from ui_walkthrough import (
    fig_averages, fig_fluxes, fig_reconstruction, fig_riemann, fig_stages, fig_update,
)

GAMMA = 1.4


def _all_figures(flux, scheme, cfg, U):
    gas, dx = cfg.gas, cfg.mesh.dx
    x = (np.arange(cfg.mesh.n_cells) + 0.5) * dx
    ts = time_step(U, scheme, cfg)
    UL, UR = interface_states(U, scheme, GAMMA, dx)
    k = default_interface(UL, UR)
    f = flux_breakdown(UL, UR, scheme, gas, dx, ts.dt)
    WL = tuple(float(v) for v in conservative_to_primitive(UL[:, k], GAMMA))
    WR = tuple(float(v) for v in conservative_to_primitive(UR[:, k], GAMMA))
    exact = np.array(conservative_to_primitive(U, GAMMA))[0]
    linear = scheme.name.startswith("MUSCL")
    return [
        fig_averages(x, dx, U, 0, GAMMA, exact),
        fig_reconstruction(x, dx, U, UL, UR, flat_cells(U, scheme, GAMMA, dx, 0), 0, GAMMA,
                           linear, None),
        fig_riemann(WL, WR, exact_waves(WL, WR, GAMMA),
                    model_waves(flux, UL[:, k], UR[:, k], gas), dx, ts.dt, 0),
        fig_fluxes(x, dx, U, f, 0, 0, GAMMA, k),
        fig_update(x, dx, U, stage_update(U, f.numerical, ts.dt, dx), f.numerical, 7, 0, GAMMA),
        fig_stages(x, dx, rk_stages(U, scheme, cfg, None, ts.dt), 0, GAMMA, ts.i_max),
    ]


@pytest.mark.parametrize("flux,name", [
    ("hllc", "muscl-hllc"), ("rusanov", "rusanov"), ("ausm+", "weno5-ausm+"),
    ("godunov", "eno2-godunov"),
])
def test_all_figures_build(flux, name):
    cfg = sod_setup(16, 0.9)
    for fig in _all_figures(flux, get_scheme(name), cfg, initial_state(cfg)[1]):
        assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_figures_from_diverged_state():
    # AUSM+ at first order diverges at the first Sod step: the page keeps
    # the last physical state and must still draw every step.
    cfg = sod_setup(16, 0.9)
    scheme = get_scheme("ausm+")
    a = advance(initial_state(cfg)[1], scheme, cfg, None, 10)
    assert a.diverged
    for fig in _all_figures("ausm+", scheme, cfg, a.U):
        assert isinstance(fig, go.Figure)


def test_reconstruction_marks_flat_cells():
    cfg = sod_setup(16, 0.9)
    scheme = get_scheme("muscl-hllc", limiter="minmod")
    U = initial_state(cfg)[1]
    dx = cfg.mesh.dx
    x = (np.arange(16) + 0.5) * dx
    UL, UR = interface_states(U, scheme, GAMMA, dx)
    fig = fig_reconstruction(x, dx, U, UL, UR, flat_cells(U, scheme, GAMMA, dx, 0), 0, GAMMA,
                             True, None)
    assert any(t.name == "pente annulée" for t in fig.data)


def test_cell_content_shaded_only_for_density():
    cfg = sod_setup(16, 0.9)
    U = initial_state(cfg)[1]
    x = (np.arange(16) + 0.5) * cfg.mesh.dx
    names = lambda fig: [t.name for t in fig.data]
    assert "contenu de la maille" in names(fig_averages(x, cfg.mesh.dx, U, 0, GAMMA, None))
    assert "contenu de la maille" not in names(fig_averages(x, cfg.mesh.dx, U, 1, GAMMA, None))


def test_riemann_diagram_shows_every_wave_within_one_step():
    # At the Sod start-up the exact shock (1.75) is faster than the cell
    # speeds used for dt (1.18): it travels 1.33 dx in one step.
    WL, WR = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)
    dx, dt = 1 / 16, 0.9 / 16 / np.sqrt(1.4)
    waves = exact_waves(WL, WR, GAMMA)
    fig = fig_riemann(WL, WR, waves, None, dx, dt, 0)
    fastest = max(abs(s) for w in waves for s in w.speeds) * dt
    assert fig.layout.xaxis2.range[1] >= fastest
