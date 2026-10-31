"""A failed or patched run must be reported, never returned silently."""

from dataclasses import replace

import numpy as np
import pytest

from euler1d.physics import conservative_to_primitive
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import near_vacuum, sod_shock_tube, two_shocks


def test_clean_run_reports_ok():
    result = run_simulation(sod_shock_tube(100), get_scheme("hllc"))
    assert result.status == "ok"
    assert result.floor_events == 0
    assert result.floor_energy == 0.0


@pytest.mark.filterwarnings("ignore:Positivity floor")
def test_divergence_is_reported_with_last_finite_state():
    with pytest.warns(RuntimeWarning, match="diverg"):
        # An unlimited centred scheme cannot survive two strong shocks.
        result = run_simulation(two_shocks(200), get_scheme("lax-wendroff"))
    assert result.status == "diverged"
    assert np.isfinite(result.times[-1])
    assert result.times[-1] < result.config.time.t_final
    U = result.snapshots[-1]
    rho, u, p = conservative_to_primitive(U, 1.4)
    assert (rho > 0).all() and (p > 0).all()


@pytest.mark.filterwarnings("ignore:Solution diverged")
def test_floor_is_off_by_default_so_the_scheme_shows_its_own_failure():
    cfg = replace(near_vacuum(400), bc_type="reflective")
    result = run_simulation(cfg, get_scheme("roe-nc"))
    assert result.floor_events == 0
    assert result.status == "diverged"


def test_positivity_floor_is_reported():
    cfg = replace(near_vacuum(400), bc_type="reflective")
    with pytest.warns(RuntimeWarning, match="floor"):
        result = run_simulation(cfg, get_scheme("roe-nc"), positivity_floor=True)
    assert result.status == "ok"
    assert result.floor_events > 0
    assert result.floor_energy > 0.0
