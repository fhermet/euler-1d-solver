"""Scalar advection of a square wave (chapter 1 of the walkthrough)."""

import numpy as np
import pytest

from euler1d.advection import PHI, advect, cut_cell, exact_averages


@pytest.mark.parametrize("n_cells", [10, 40, 80])
@pytest.mark.parametrize("t", [0.0, 0.3])
def test_averages_hold_the_square_width(n_cells, t):
    assert exact_averages(n_cells, t).sum() / n_cells == pytest.approx(0.26, abs=1e-12)


@pytest.mark.parametrize("n_cells", [10, 40, 80])
def test_cut_cell_is_intermediate(n_cells):
    assert 0 < exact_averages(n_cells, 0.0)[cut_cell(n_cells)] < 1


def test_upwind_at_cfl_one_is_exact():
    u0 = exact_averages(40, 0.0)
    np.testing.assert_allclose(advect(u0, "upwind", 1.0, 7).u, np.roll(u0, 7), atol=1e-12)


def test_centred_grows():
    run = advect(exact_averages(40, 0.0), "centred", 0.8, 40)
    assert run.exploded or run.u.max() > 1.05


@pytest.mark.parametrize("limiter", ["minmod", "van-leer"])
def test_limited_slopes_stay_in_bounds(limiter):
    u = advect(exact_averages(40, 0.0), "muscl", 0.5, 80, PHI[limiter]).u
    assert u.min() >= -1e-12 and u.max() <= 1 + 1e-12


def test_unlimited_slope_overshoots():
    assert advect(exact_averages(40, 0.0), "muscl", 0.5, 80, PHI["fromm"]).u.max() > 1.01


def test_explosion_above_cfl_one():
    run = advect(exact_averages(40, 0.0), "upwind", 1.1, 40)
    assert run.exploded and run.steps_done < 40
