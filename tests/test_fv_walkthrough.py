"""Computations of the finite-volume walkthrough page."""

import numpy as np
import pytest

from euler1d.boundary import apply_boundary_conditions
from euler1d.fv_walkthrough import (
    advance, default_interface, flat_cells, flux_breakdown, global_balance,
    interface_states, rk_stages, sod_setup, stage_update, time_step,
)
from euler1d.schemes import get_scheme
from euler1d.solver import _compute_rhs, _rk_step, initial_state, run_simulation


def _setup(name, n_cells=16, cfl=0.9):
    cfg = sod_setup(n_cells, cfl)
    return cfg, get_scheme(name), initial_state(cfg)[1]


def test_balance_equals_boundary_fluxes():
    cfg, scheme, U = _setup("muscl-hllc")
    gas, dx = cfg.gas, cfg.mesh.dx
    ts = time_step(U, scheme, cfg)
    f = flux_breakdown(*interface_states(U, scheme, gas.gamma, dx), scheme, gas, dx, ts.dt)
    b = global_balance(U, stage_update(U, f.numerical, ts.dt, dx), f.numerical, ts.dt, dx)
    np.testing.assert_allclose(b.change, b.boundary, rtol=0, atol=1e-13)
    assert b.change[0] == pytest.approx(0.0, abs=1e-14)    # mass: u = 0 at both ends
    assert abs(b.change[1]) > 1e-3                         # momentum: p = 1 and 0.1


def test_flux_is_centred_minus_dissipation():
    cfg, scheme, U = _setup("muscl-hll")
    UL, UR = interface_states(U, scheme, cfg.gas.gamma, cfg.mesh.dx)
    f = flux_breakdown(UL, UR, scheme, cfg.gas, cfg.mesh.dx, 1e-3)
    np.testing.assert_allclose(f.centred - f.dissipation, f.numerical, atol=1e-14)


@pytest.mark.parametrize("name", ["muscl-hllc", "weno5-hll", "hllc"])
def test_interface_states_are_those_of_the_scheme(name):
    cfg, scheme, U = _setup(name)
    gas, dx = cfg.gas, cfg.mesh.dx
    UL, UR = interface_states(U, scheme, gas.gamma, dx)
    U_ext = apply_boundary_conditions(U, "transmissive", scheme.n_ghost)
    np.testing.assert_array_equal(scheme.compute_riemann_flux(UL, UR, gas, dx, 1e-3),
                                  scheme.compute_fluxes(U_ext, gas, dx, 1e-3))


@pytest.mark.parametrize("rk,evaluations", [("RK1", 1), ("RK2", 2), ("RK3", 3), ("RK4", 4), ("RK5", 6)])
def test_rk_stage_counts(rk, evaluations):
    cfg, scheme, U = _setup("muscl-hllc")
    dt = time_step(U, scheme, cfg).dt
    stages = rk_stages(U, scheme, cfg, rk, dt)
    assert len(stages) - 1 == evaluations
    rhs = lambda v: _compute_rhs(v, scheme, cfg.gas, cfg.mesh.dx, dt, "transmissive", dt)
    np.testing.assert_array_equal(stages[-1], _rk_step(U, rk, rhs))


def test_advance_matches_run_simulation():
    from dataclasses import replace
    cfg, scheme, U = _setup("muscl-hllc")
    a = advance(U, scheme, cfg, None, 20)
    assert a.steps_done == 20 and not a.diverged
    ref = run_simulation(replace(cfg, time=replace(cfg.time, t_final=a.elapsed)), scheme,
                         n_snapshots=0)
    np.testing.assert_allclose(a.U, ref.snapshots[-1], rtol=0, atol=1e-12)


def test_advance_stops_at_divergence():
    # AUSM+ at first order diverges on the Sod start-up at CFL 0.9 (docs 4.9).
    cfg, scheme, U = _setup("ausm+")
    a = advance(U, scheme, cfg, None, 10)
    assert a.diverged and a.steps_done == 0
    np.testing.assert_array_equal(a.U, U)


def test_time_step_caps_cfl():
    cfg, scheme, U = _setup("muscl-hllc")
    assert time_step(U, scheme, cfg).cfl == 0.5
    cfg, scheme, U = _setup("hllc")
    ts = time_step(U, scheme, cfg)
    assert ts.cfl == 0.9
    assert ts.i_max < 8                       # left state: largest sound speed
    assert ts.dt == pytest.approx(0.9 * cfg.mesh.dx / ts.s_max)


def test_odd_mesh_mixed_cell():
    # With 15 cells the discontinuity sits in the middle of cell 7.
    cfg, scheme, U = _setup("muscl-hllc", n_cells=15)
    UL, UR = interface_states(U, scheme, cfg.gas.gamma, cfg.mesh.dx)
    assert default_interface(UL, UR) in (7, 8)
    flat = flat_cells(U, scheme, cfg.gas.gamma, cfg.mesh.dx, 0)
    assert flat.shape == (15,) and flat.dtype == bool


from euler1d.config import GasProperties
from euler1d.fv_walkthrough import exact_waves, model_waves
from euler1d.physics import primitive_to_conservative


def test_sod_exact_waves():
    waves = exact_waves((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 1.4)
    assert [w.kind for w in waves] == ["rarefaction", "contact", "shock"]
    head, tail = waves[0].speeds
    assert head == pytest.approx(-np.sqrt(1.4), rel=1e-12)
    assert head < tail < waves[1].speeds[0]
    assert waves[1].speeds[0] == pytest.approx(0.92745, rel=1e-4)
    assert waves[2].speeds[0] == pytest.approx(1.75216, rel=1e-4)


def test_vacuum_waves():
    waves = exact_waves((1.0, -5.0, 0.4), (1.0, 5.0, 0.4), 1.4)
    assert [w.kind for w in waves] == ["rarefaction", "vacuum", "rarefaction"]


def test_model_waves_per_flux():
    gas = GasProperties(1.4)
    UL = primitive_to_conservative(1.0, 0.0, 1.0, 1.4)
    UR = primitive_to_conservative(0.125, 0.0, 0.1, 1.4)
    assert len(model_waves("hll", UL, UR, gas)) == 2
    sl, s_star, sr = model_waves("hllc", UL, UR, gas)
    assert sl < s_star < sr
    lo, hi = model_waves("rusanov", UL, UR, gas)
    assert lo == pytest.approx(-hi)
    assert len(model_waves("roe-nc", UL, UR, gas)) == 3
    assert model_waves("godunov", UL, UR, gas) == []
    assert model_waves("ausm+", UL, UR, gas) is None
    assert model_waves("ausm+-up", UL, UR, gas) is None
