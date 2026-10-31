"""Invalid configurations are rejected with a clear error."""

from dataclasses import replace

import pytest

from euler1d.config import (
    GasProperties, MeshConfig, PrimitiveState, RiemannProblem, TimeConfig,
)
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import sod_shock_tube


@pytest.mark.parametrize("make", [
    lambda: GasProperties(gamma=1.0),
    lambda: MeshConfig(0.0, 1.0, 0),
    lambda: MeshConfig(1.0, 0.0, 10),
    lambda: TimeConfig(t_final=0.0),
    lambda: TimeConfig(t_final=1.0, cfl=0.0),
    lambda: TimeConfig(t_final=1.0, dt_fixed=-1e-3),
    lambda: PrimitiveState(rho=0.0, u=0.0, p=1.0),
    lambda: PrimitiveState(rho=1.0, u=0.0, p=-1.0),
])
def test_invalid_parameters_raise(make):
    with pytest.raises(ValueError):
        make()


def test_discontinuity_outside_domain_raises():
    cfg = sod_shock_tube(50)
    with pytest.raises(ValueError, match="discontinuity"):
        replace(cfg, problem=replace(cfg.problem, x_discontinuity=2.0))


def test_unknown_boundary_condition_raises():
    with pytest.raises(ValueError, match="boundary"):
        replace(sod_shock_tube(50), bc_type="outflow")


def test_mesh_smaller_than_stencil_raises():
    cfg = replace(sod_shock_tube(50), mesh=MeshConfig(0.0, 1.0, 2))
    with pytest.raises(ValueError, match="ghost"):
        run_simulation(cfg, get_scheme("weno5-hllc"))


@pytest.mark.filterwarnings("ignore:Solution diverged")
def test_fixed_step_above_cfl_limit_warns():
    cfg = sod_shock_tube(100)
    cfg = replace(cfg, time=replace(cfg.time, dt_fixed=0.05, t_final=0.05))
    with pytest.warns(RuntimeWarning, match="CFL"):
        run_simulation(cfg, get_scheme("rusanov"))
