"""Time-stepping details that must not change the physics."""

from dataclasses import replace

import numpy as np

from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import sod_shock_tube


def test_lax_friedrichs_tiny_final_step_does_not_smooth_the_solution():
    # A final step of 1e-9 must move the solution by O(1e-9), not apply a
    # full Lax-Friedrichs averaging.
    base = sod_shock_tube(200)
    cfg = replace(base, time=replace(base.time, t_final=0.1, dt_fixed=1e-3))
    longer = replace(cfg, time=replace(cfg.time, t_final=0.1 + 1e-9))
    a = run_simulation(cfg, get_scheme("lax-friedrichs")).snapshots[-1]
    b = run_simulation(longer, get_scheme("lax-friedrichs")).snapshots[-1]
    assert np.abs(a - b).max() < 1e-6


import pytest

from euler1d.test_cases import shu_osher
import euler1d.solver as solver_module


@pytest.mark.parametrize("name,expected", [
    ("hllc", 0.9), ("lax-friedrichs", 0.9), ("muscl-hllc", 0.5),
    ("eno2-roe", 0.5), ("weno5-hllc", 0.5),
])
def test_cfl_is_capped_by_the_scheme_limit(name, expected):
    result = run_simulation(sod_shock_tube(50), get_scheme(name), n_snapshots=0)
    assert result.cfl == expected


def test_muscl_on_shu_osher_does_not_amplify_round_off(monkeypatch):
    # Above CFL 0.5, MUSCL + RK2 is not TVD: a 1e-14 perturbation of the
    # initial state grew to 0.4 at CFL 0.9 with minmod.
    scheme = get_scheme("muscl-hllc", limiter="minmod")
    a = run_simulation(shu_osher(400), scheme, n_snapshots=0).snapshots[-1]
    original = solver_module.initial_state
    monkeypatch.setattr(solver_module, "initial_state",
                        lambda cfg: (lambda x, U: (x, U * (1 + 1e-14)))(*original(cfg)))
    b = run_simulation(shu_osher(400), scheme, n_snapshots=0).snapshots[-1]
    assert np.abs(a - b).max() < 1e-9


def test_fixed_step_above_scheme_limit_warns():
    base = sod_shock_tube(100)
    # CFL about 0.7 for the initial state: fine for HLLC, too large for MUSCL.
    cfg = replace(base, time=replace(base.time, dt_fixed=5.9e-3, t_final=1.18e-2))
    with pytest.warns(RuntimeWarning, match="CFL"):
        run_simulation(cfg, get_scheme("muscl-hllc"))
