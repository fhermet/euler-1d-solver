"""Finite-volume schemes must conserve mass, momentum and energy exactly."""

import numpy as np
import pytest

from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import entropy_wave

SCHEMES = ["rusanov", "hll", "hllc", "roe", "ausm+", "jst",
           "muscl-hllc", "eno2-roe", "weno3-hll", "wenoz5-hllc"]


@pytest.mark.parametrize("name", SCHEMES)
def test_periodic_totals_are_conserved(name):
    cfg = entropy_wave(n_cells=50)
    result = run_simulation(cfg, get_scheme(name), n_snapshots=0)
    x = result.x
    rho, u, p = cfg.problem.init_fn(x, cfg.gas.gamma)
    U0 = np.array([rho, rho * u, p / (cfg.gas.gamma - 1.0) + 0.5 * rho * u**2])
    np.testing.assert_allclose(result.snapshots[-1].sum(axis=1), U0.sum(axis=1),
                               rtol=1e-12)
