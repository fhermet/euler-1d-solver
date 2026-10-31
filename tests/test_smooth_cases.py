"""Smooth test cases: cell-average initialisation and observed orders."""

import numpy as np
import pytest

from euler1d.physics import cell_average
from euler1d.results import compute_errors, compute_exact_solution
from euler1d.schemes import get_scheme
from euler1d.solver import initial_state, run_simulation
from euler1d.test_cases import entropy_wave, isentropic_wave

GAMMA = 1.4


def test_cell_average_is_exact_for_polynomials():
    # 5-point Gauss is exact up to degree 9.
    x, dx = np.array([0.3, 0.7]), 0.2
    U = cell_average(lambda s: (1 + s**4, 0 * s, 1 + 0 * s), x, dx, GAMMA)
    exact = 1 + ((x + dx / 2) ** 5 - (x - dx / 2) ** 5) / (5 * dx)
    np.testing.assert_allclose(U[0], exact, rtol=1e-14)


def test_initial_state_holds_cell_averages():
    cfg = entropy_wave(20)
    x, U0 = initial_state(cfg)
    expected = cell_average(lambda s: cfg.problem.init_fn(s, GAMMA), x, cfg.mesh.dx, GAMMA)
    np.testing.assert_allclose(U0, expected, rtol=1e-14)
    # rho = 1 + 0.2 sin(2 pi x): the mass over one period is exactly 1.
    assert U0[0].sum() * cfg.mesh.dx == pytest.approx(1.0, rel=1e-14)


def test_isentropic_wave_starts_from_its_initial_condition():
    cfg = isentropic_wave(64)
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(cfg.problem.exact_fn(x, 0.0, GAMMA),
                               cfg.problem.init_fn(x, GAMMA), rtol=1e-13)


def test_isentropic_wave_is_a_simple_wave():
    # The left-going Riemann invariant u - 2a/(gamma-1) stays uniform.
    cfg = isentropic_wave(64)
    x = np.linspace(0, 1, 101)
    rho, u, p = cfg.problem.exact_fn(x, cfg.time.t_final, GAMMA)
    J = u - 2.0 * np.sqrt(GAMMA * p / rho) / (GAMMA - 1.0)
    np.testing.assert_allclose(J, J[0], rtol=1e-12)
    np.testing.assert_allclose(p, rho**GAMMA, rtol=1e-12)


def _observed_order(name, variables, Ns=(40, 80, 160)):
    errors = []
    for N in Ns:
        cfg = isentropic_wave(N)
        result = run_simulation(cfg, get_scheme(name, variables=variables),
                                time_integrator="RK5")
        errors.append(compute_errors(result, compute_exact_solution(cfg))["rho"]["L1"])
    return np.log2(errors[-2] / errors[-1])


def test_weno5_order_on_isentropic_wave_depends_on_variables():
    assert _observed_order("wenoz5-hllc", "primitive") < 3.0
    assert _observed_order("wenoz5-hllc", "characteristic") > 4.5


def test_riemann_cell_holding_the_discontinuity_mixes_both_states():
    from dataclasses import replace
    from euler1d.config import MeshConfig
    from euler1d.test_cases import sod_shock_tube
    cfg = replace(sod_shock_tube(), mesh=MeshConfig(0.0, 1.0, 5))  # x_d = 0.5 mid-cell 2
    _, U0 = initial_state(cfg)
    assert U0[0, 2] == pytest.approx(0.5 * (1.0 + 0.125))
    assert U0[0, 1] == 1.0 and U0[0, 3] == 0.125


def test_shu_osher_reference_depends_on_the_requested_time():
    # The reference was cached once, whatever the time asked for.
    from euler1d.test_cases import shu_osher
    exact_fn = shu_osher(100).problem.exact_fn
    x = np.linspace(-4.5, 4.5, 50)
    early = exact_fn(x, 0.1, GAMMA)
    later = exact_fn(x, 0.2, GAMMA)
    assert not np.allclose(early[0], later[0])
    np.testing.assert_array_equal(exact_fn(x, 0.1, GAMMA)[0], early[0])


def test_shu_osher_reference_at_time_zero_is_the_initial_condition():
    from euler1d.test_cases import shu_osher
    prob = shu_osher(100).problem
    x = np.linspace(-4.5, 4.5, 50)
    np.testing.assert_array_equal(prob.exact_fn(x, 0.0, GAMMA)[0], prob.init_fn(x, GAMMA)[0])
