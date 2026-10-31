"""Boundary conditions."""

import numpy as np
import pytest

from euler1d.config import GasProperties, MeshConfig, SimulationConfig, SmoothProblem, TimeConfig
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation


def _initial(x):
    rho = 1 + 0.5 * np.exp(-100 * (x - 0.3) ** 2)
    u = 0.5 * np.sin(np.pi * x) ** 2
    p = 1 + 0.3 * np.exp(-100 * (x - 0.6) ** 2)
    return rho, u, p


def _mirrored(x):
    """Initial data on [0, 2], symmetric about x = 1 with u -> -u."""
    rho, u, p = _initial(np.where(x <= 1, x, 2 - x))
    return rho, np.where(x <= 1, u, -u), p


def _config(fn, length, n, bc):
    return SimulationConfig(GasProperties(1.4), MeshConfig(0.0, length, n), TimeConfig(0.8),
                            SmoothProblem(lambda x, g: fn(x), lambda x, t, g: fn(x)), bc)


@pytest.mark.parametrize("name", ["hllc", "roe", "lax-wendroff", "jst", "muscl-hllc",
                                  "eno2-hll", "weno5-hllc", "wenoz5-roe"])
def test_reflective_walls_equal_the_mirrored_periodic_problem(name):
    # Walls at x = 0 and x = 1 act as symmetry planes: the solution must be
    # the restriction of the periodic solution on [0, 2] with mirrored data.
    walls = run_simulation(_config(_initial, 1.0, 100, "reflective"), get_scheme(name), n_snapshots=0)
    mirror = run_simulation(_config(_mirrored, 2.0, 200, "periodic"), get_scheme(name), n_snapshots=0)
    np.testing.assert_allclose(walls.snapshots[-1], mirror.snapshots[-1][:, :100], atol=1e-11)
