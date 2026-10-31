"""Exact Riemann solver against Toro (2009), Table 4.2."""

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
