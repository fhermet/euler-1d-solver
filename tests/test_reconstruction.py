"""Reconstruction schemes on strong-wave cases and basic stencil properties."""

import numpy as np
import pytest

from euler1d.physics import primitive_to_conservative
from euler1d.schemes import get_scheme
from euler1d.schemes.reconstruction.eno import eno_reconstruct
from euler1d.solver import run_simulation
from euler1d.test_cases import (
    double_rarefaction, near_vacuum, stationary_contact, two_shocks,
)


def test_eno2_picks_the_smaller_slope():
    # Cell 2 sits between slopes 1 (left) and 10 (right): ENO2 takes 1.
    rho = np.array([1.0, 2.0, 3.0, 13.0, 23.0])
    U = primitive_to_conservative(rho, np.zeros(5), np.ones(5), 1.4)
    UL, _ = eno_reconstruct(U, 1.4)
    assert UL[0, 1] == pytest.approx(3.5)


@pytest.mark.parametrize("case", [two_shocks, double_rarefaction])
@pytest.mark.parametrize("flux", ["rusanov", "hllc", "roe"])
def test_eno2_runs_strong_cases_without_floor(case, flux):
    result = run_simulation(case(200), get_scheme(f"eno2-{flux}"), n_snapshots=0)
    assert result.status == "ok"
    assert result.floor_events == 0


@pytest.mark.parametrize("name", ["weno5-rusanov", "wenoz5-rusanov", "weno5-hll"])
def test_weno5_default_integrator_keeps_contact_stable(name):
    result = run_simulation(stationary_contact(200), get_scheme(name), n_snapshots=0)
    assert result.status == "ok"
    assert result.floor_events == 0


@pytest.mark.parametrize("case", [double_rarefaction, near_vacuum])
@pytest.mark.parametrize("name", ["weno5-hllc", "wenoz5-hll", "weno3-hll"])
def test_high_order_runs_near_vacuum_cases_at_their_default_cfl(case, name):
    # Roe is left out: it is not positively conservative (Einfeldt et al.,
    # 1991) and wenoz5-roe still fails on near_vacuum at CFL 0.5.
    result = run_simulation(case(200), get_scheme(name), n_snapshots=0)
    assert result.status == "ok"


def test_roe_without_entropy_fix_shows_an_expansion_shock():
    # Toro's test 1: the left rarefaction is transonic. Without entropy fix,
    # Roe keeps a jump inside it (an expansion shock), which violates the
    # entropy condition.
    from euler1d.results import compute_exact_solution
    from euler1d.test_cases import sonic_rarefaction
    cfg = sonic_rarefaction(100)
    exact = compute_exact_solution(cfg)["rho"].values
    jump = lambda rho: np.abs(np.diff(rho[:50])).max()   # left half: the fan
    no_fix = run_simulation(cfg, get_scheme("roe-nc"), n_snapshots=0).snapshots[-1][0]
    fixed = run_simulation(cfg, get_scheme("roe"), n_snapshots=0).snapshots[-1][0]
    assert jump(no_fix) > 2.5 * jump(fixed)
    assert jump(fixed) < 1.5 * jump(exact)
    # A genuine discontinuity: the jump does not shrink with the mesh.
    fine = sonic_rarefaction(400)
    no_fix_fine = run_simulation(fine, get_scheme("roe-nc"), n_snapshots=0).snapshots[-1][0]
    assert np.abs(np.diff(no_fix_fine[:200])).max() > 0.1
