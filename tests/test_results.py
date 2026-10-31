"""Post-processing: convergence orders."""

import numpy as np

from euler1d.results import convergence_study, errors_at_round_off
from euler1d.schemes import get_scheme
from euler1d.test_cases import entropy_wave, stationary_contact


def _errors(case, name, var="rho"):
    df = convergence_study(case, get_scheme(name), [25, 50, 100])
    return df[df["variable"] == var]["L1"].values


def test_round_off_errors_are_detected():
    # HLLC keeps a stationary contact exactly; u and p are constant in the
    # entropy wave. No order can be measured from round-off.
    assert errors_at_round_off(_errors(stationary_contact, "hllc"))
    assert errors_at_round_off(_errors(entropy_wave, "weno5-hllc", "p"))


def test_genuine_errors_are_not_flagged():
    assert not errors_at_round_off(_errors(stationary_contact, "hll"))
    assert not errors_at_round_off(_errors(entropy_wave, "weno5-hllc", "rho"))


import pytest

from euler1d.results import expected_order, local_orders
from euler1d.test_cases import isentropic_wave, sod_shock_tube


@pytest.mark.parametrize("name,kwargs,rk,case,expected", [
    ("hllc", {}, None, entropy_wave, 1),
    ("muscl-hllc", {}, None, entropy_wave, 2),
    ("weno5-hllc", {}, None, entropy_wave, 3),           # SSP-RK3 limits it
    ("weno5-hllc", {}, "RK5", entropy_wave, 5),
    ("weno5-hllc", {}, "RK5", isentropic_wave, 2),       # primitive variables
    ("weno5-hllc", {"variables": "characteristic"}, "RK5", isentropic_wave, 5),
    ("lax-wendroff", {}, None, entropy_wave, 2),         # time step in the flux
    ("jst", {}, None, entropy_wave, 2),
])
def test_expected_order_on_smooth_cases(name, kwargs, rk, case, expected):
    order, _ = expected_order(get_scheme(name, **kwargs), rk, case(50))
    assert order == expected


def test_no_single_expected_order_for_discontinuous_solutions():
    order, note = expected_order(get_scheme("hllc"), None, sod_shock_tube(50))
    assert order is None and "contact" in note


def test_local_orders_between_successive_meshes():
    dx = np.array([0.1, 0.05, 0.025])
    err = np.array([1.0, 0.25, 0.125])
    np.testing.assert_allclose(local_orders(dx, err), [2.0, 1.0])
