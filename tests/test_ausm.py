"""AUSM+ (Liou 1996) and AUSM+-up (Liou 2006)."""

from dataclasses import replace

import numpy as np
import pytest

from euler1d import test_cases as tc
from euler1d.config import GasProperties
from euler1d.physics import primitive_to_conservative
from euler1d.schemes import get_scheme
from euler1d.schemes.flux import AUSMPlusUp
from euler1d.solver import run_simulation

ALL_CASES = [tc.sod_shock_tube, tc.lax_test, tc.double_rarefaction,
             tc.stationary_contact, tc.near_vacuum, tc.two_shocks, tc.shu_osher,
             tc.entropy_wave, tc.acoustic_wave]


def test_ausm_up_mass_flux_for_pressure_jump_at_rest():
    # u = 0 on both sides: only the pressure-diffusion term of Liou (2006)
    # drives mass, m = -(Kp/fa) * dp / a_half, with a_half = min(a*_L, a*_R).
    g, M_inf = 1.4, 0.5
    pL, pR = 1.0, 1.1
    UL = primitive_to_conservative(np.array([1.0]), np.array([0.0]), np.array([pL]), g)
    UR = primitive_to_conservative(np.array([1.0]), np.array([0.0]), np.array([pR]), g)
    F = AUSMPlusUp(M_inf=M_inf).compute_riemann_flux(UL, UR, GasProperties(g), 1.0, 1.0)

    a_star = lambda p: np.sqrt(2.0 * (g - 1.0) / (g + 1.0) * g / (g - 1.0) * p)
    a_half = min(a_star(pL), a_star(pR))
    fa = M_inf * (2.0 - M_inf)
    assert F[0, 0] == pytest.approx(-(0.25 / fa) * (pR - pL) / a_half, rel=1e-12)


@pytest.mark.parametrize("name", ["ausm+", "ausm+-up"])
def test_stationary_contact_is_preserved_exactly(name):
    cfg = tc.stationary_contact(100)
    result = run_simulation(cfg, get_scheme(name), n_snapshots=0)
    x = result.x
    rho0 = np.where(x < cfg.problem.x_discontinuity, 1.0, 0.125)
    np.testing.assert_allclose(result.snapshots[-1][0], rho0, rtol=1e-12)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("name", ["ausm+-up", "muscl-ausm+-up"])
def test_ausm_up_runs_every_case(case, name):
    result = run_simulation(case(200), get_scheme(name), n_snapshots=0)
    assert result.status == "ok"


@pytest.mark.filterwarnings("ignore:Solution diverged")
def test_ausm_plus_needs_a_smaller_cfl_on_sod_startup():
    # No pressure diffusion in the AUSM+ mass flux: at the initial jump
    # (u = 0) the low-density cell receives momentum but no mass.
    cfg = tc.sod_shock_tube(200)
    assert run_simulation(cfg, get_scheme("ausm+")).status == "diverged"
    cfg = replace(cfg, time=replace(cfg.time, cfl=0.5))
    assert run_simulation(cfg, get_scheme("ausm+")).status == "ok"
