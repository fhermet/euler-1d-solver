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
