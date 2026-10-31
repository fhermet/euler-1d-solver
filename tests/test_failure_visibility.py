"""A failed or patched run must be reported, never returned silently."""

from dataclasses import replace

import numpy as np
import pytest

from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import near_vacuum, sod_shock_tube


def test_clean_run_reports_ok():
    result = run_simulation(sod_shock_tube(100), get_scheme("hllc"))
    assert result.status == "ok"
    assert result.floor_events == 0
    assert result.floor_energy == 0.0


@pytest.mark.filterwarnings("ignore:Positivity floor")
def test_divergence_is_reported_with_last_finite_state():
    with pytest.warns(RuntimeWarning, match="diverg"):
        result = run_simulation(near_vacuum(400), get_scheme("weno5-roe"))
    assert result.status == "diverged"
    assert np.isfinite(result.times[-1])
    assert result.times[-1] < result.config.time.t_final
    assert np.isfinite(result.snapshots[-1]).all()


def test_positivity_floor_is_reported():
    cfg = replace(near_vacuum(400), bc_type="reflective")
    with pytest.warns(RuntimeWarning, match="floor"):
        result = run_simulation(cfg, get_scheme("roe-nc"))
    assert result.status == "ok"
    assert result.floor_events > 0
    assert result.floor_energy > 0.0
