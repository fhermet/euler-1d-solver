"""Figures of the eight walkthrough screens."""

import numpy as np
import plotly.graph_objects as go
import pytest

from euler1d.advection import PHI, advect, exact_averages
from ui_walkthrough import (
    CFL_SCREEN_4, fig_screen_1, fig_screen_2, fig_screen_3, fig_screen_4, fig_screen_5,
    fig_screen_6, fig_screen_7, fig_screen_8,
)

U0 = exact_averages(40, 0.0)


def _bubbles(fig: go.Figure):
    return [a for a in fig.layout.annotations if a.showarrow]


def _figures():
    x = (np.arange(100) + 0.5) / 100
    rho = np.where(x < 0.5, 1.0, 0.125)
    return {
        1: fig_screen_1(0.3),
        2: fig_screen_2(10),
        3: fig_screen_3(U0, advect(U0, "upwind", 0.8, 1).u, 4, 0.8),
        4: fig_screen_4(advect(U0, "centred", CFL_SCREEN_4, 10), "centred", 10),
        5: fig_screen_5(advect(U0, "upwind", 1.0, 10), 1.0, 10),
        6: fig_screen_6(advect(U0, "muscl", 0.5, 40, PHI["minmod"]).u, "minmod", 0.5),
        7: fig_screen_7("hll"),
        8: fig_screen_8(x, rho, "hllc", 3),
    }


@pytest.mark.parametrize("screen", range(1, 9))
def test_every_screen_has_one_bubble(screen):
    fig = _figures()[screen]
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1
    assert len(_bubbles(fig)) == 1


@pytest.mark.parametrize("choice", ["exact", "hll", "hllc"])
def test_screen_7_each_choice(choice):
    assert len(_bubbles(fig_screen_7(choice))) == 1


def test_exploded_run_is_readable():
    run = advect(U0, "upwind", 1.3, 40)
    assert run.exploded
    fig = fig_screen_5(run, 1.3, 40)
    text = _bubbles(fig)[0].text
    assert "explose" in text and str(run.steps_done + 1) in text
    assert list(fig.layout.yaxis.range) == [-0.5, 1.5]


@pytest.mark.parametrize("make", [
    lambda: fig_screen_4(advect(U0, "upwind", CFL_SCREEN_4, 0), "upwind", 0),
    lambda: fig_screen_5(advect(U0, "upwind", 0.5, 0), 0.5, 0),
])
def test_zero_steps_shows_initial_state(make):
    fig = make()
    assert len(_bubbles(fig)) == 1
