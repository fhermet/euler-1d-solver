"""Exact Riemann solver against the tests of Toro (2009), §4.3.3."""

import pytest

from euler1d.riemann import solve_riemann

GAMMA = 1.4

# (left state, right state, p*, u*, rho*_L, rho*_R)
TORO_TABLE_4_2 = [
    ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.30313, 0.92745, 0.42632, 0.26557),
    ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.00189, 0.00000, 0.02185, 0.02185),
    ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 460.894, 19.5975, 0.57506, 5.99924),
    ((1.0, 0.0, 0.01), (1.0, 0.0, 100.0), 46.0950, -6.19633, 5.99242, 0.57511),
    ((5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.0950),
     1691.64, 8.68975, 14.2823, 31.0426),
]


@pytest.mark.parametrize("left,right,p,u,rl,rr", TORO_TABLE_4_2)
def test_star_region_matches_toro(left, right, p, u, rl, rr):
    pstar, ustar, rhoL, rhoR = solve_riemann(*left, *right, GAMMA)
    assert pstar == pytest.approx(p, rel=1e-4, abs=1e-5)
    assert ustar == pytest.approx(u, rel=1e-4, abs=1e-5)
    assert rhoL == pytest.approx(rl, rel=1e-4)
    assert rhoR == pytest.approx(rr, rel=1e-4)


import numpy as np

from euler1d.riemann import sample_interfaces, sample_riemann


def _shifted_sample(xi, left, right):
    """Galilean invariance: the state at x/t = xi is the state at 0 of the
    problem seen from a frame moving at speed xi."""
    n = len(xi)
    rho, u, p = sample_interfaces(
        np.full(n, left[0]), left[1] - xi, np.full(n, left[2]),
        np.full(n, right[0]), right[1] - xi, np.full(n, right[2]), GAMMA)
    return rho, u + xi, p


@pytest.mark.parametrize("left,right", [row[:2] for row in TORO_TABLE_4_2])
def test_sampling_agrees_with_the_interface_solver(left, right):
    xi = np.linspace(-25.0, 25.0, 401)
    expected = _shifted_sample(xi, left, right)
    for got, exp in zip(sample_riemann(xi, 1.0, *left, *right, GAMMA), expected):
        np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-12)


def test_sampling_through_a_generated_vacuum():
    # u_R - u_L = 10 > 2 (a_L + a_R) / (gamma - 1): two rarefactions and a
    # vacuum (Toro 2009, 4.6.3). Used to raise ValueError.
    left, right = (1.0, -5.0, 0.4), (1.0, 5.0, 0.4)
    xi = np.linspace(-8.0, 8.0, 321)
    rho, u, p = sample_riemann(xi, 1.0, *left, *right, GAMMA)
    a = np.sqrt(GAMMA * 0.4)
    tail = -5.0 + 2.0 * a / (GAMMA - 1.0)       # end of the left fan
    inside = np.abs(xi) < tail
    assert (rho[inside] == 0).all() and (p[inside] == 0).all()
    np.testing.assert_allclose(rho, rho[::-1], atol=1e-14)   # symmetric problem
    for got, exp in zip((rho, u, p), _shifted_sample(xi, left, right)):
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-14)
