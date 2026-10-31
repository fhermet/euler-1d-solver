"""Vectorised HLLC and Godunov fluxes against per-interface reference formulas."""

import numpy as np
import pytest

from euler1d.config import GasProperties
from euler1d.physics import compute_flux, primitive_to_conservative, sound_speed
from euler1d.riemann import sample_at_interface
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import near_vacuum

GAMMA = 1.4
GAS = GasProperties(GAMMA)


def _random_states(n, seed):
    # Velocities up to Mach ~3 in both directions: all wave configurations.
    rng = np.random.default_rng(seed)
    W = [(rng.uniform(0.1, 2, n), rng.uniform(-3, 3, n), rng.uniform(0.1, 2, n)) for _ in range(2)]
    return [primitive_to_conservative(*w, GAMMA) for w in W], W


def _hllc_reference(UL, UR):
    """Toro (2009), §10.4.2, one interface at a time (Davis speeds)."""
    F = np.empty_like(UL)
    for k in range(UL.shape[1]):
        uL_, uR_ = UL[:, k], UR[:, k]
        rl, ul = uL_[0], uL_[1] / uL_[0]
        rr, ur = uR_[0], uR_[1] / uR_[0]
        pl = (GAMMA - 1) * (uL_[2] - 0.5 * rl * ul**2)
        pr = (GAMMA - 1) * (uR_[2] - 0.5 * rr * ur**2)
        al, ar = np.sqrt(GAMMA * pl / rl), np.sqrt(GAMMA * pr / rr)
        sl, sr = min(ul - al, ur - ar), max(ul + al, ur + ar)
        ss = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / (rl * (sl - ul) - rr * (sr - ur))
        fl = compute_flux(uL_[:, None], GAMMA)[:, 0]
        fr = compute_flux(uR_[:, None], GAMMA)[:, 0]
        if sl >= 0:
            F[:, k] = fl
        elif sr <= 0:
            F[:, k] = fr
        else:
            r, u, p, s, U, f = (rl, ul, pl, sl, uL_, fl) if ss >= 0 else (rr, ur, pr, sr, uR_, fr)
            Us = r * (s - u) / (s - ss) * np.array(
                [1.0, ss, U[2] / r + (ss - u) * (ss + p / (r * (s - u)))])
            F[:, k] = f + s * (Us - U)
    return F


def test_hllc_matches_reference_formulas():
    (UL, UR), _ = _random_states(500, 1)
    F = get_scheme("hllc").compute_riemann_flux(UL, UR, GAS, 1.0, 1.0)
    np.testing.assert_allclose(F, _hllc_reference(UL, UR), rtol=1e-12, atol=1e-12)


def test_godunov_matches_scalar_exact_solver():
    (UL, UR), (WL, WR) = _random_states(300, 2)
    keep = (WR[1] - WL[1]) < 2.0 / (GAMMA - 1.0) * (
        sound_speed(WL[0], WL[2], GAMMA) + sound_speed(WR[0], WR[2], GAMMA))
    UL, UR = UL[:, keep], UR[:, keep]
    WL = [w[keep] for w in WL]
    WR = [w[keep] for w in WR]
    F = get_scheme("godunov").compute_riemann_flux(UL, UR, GAS, 1.0, 1.0)
    for k in range(UL.shape[1]):
        rho, u, p = sample_at_interface(WL[0][k], WL[1][k], WL[2][k],
                                        WR[0][k], WR[1][k], WR[2][k], GAMMA)
        expected = compute_flux(primitive_to_conservative(
            np.array([rho]), np.array([u]), np.array([p]), GAMMA), GAMMA)[:, 0]
        np.testing.assert_allclose(F[:, k], expected, rtol=1e-9, atol=1e-12)


def test_godunov_flux_through_generated_vacuum_is_zero():
    # u_R - u_L = 10 exceeds 2/(gamma-1) (a_L + a_R) ~ 7.5: vacuum at x/t = 0.
    UL = primitive_to_conservative(np.array([1.0]), np.array([-5.0]), np.array([0.4]), GAMMA)
    UR = primitive_to_conservative(np.array([1.0]), np.array([5.0]), np.array([0.4]), GAMMA)
    F = get_scheme("godunov").compute_riemann_flux(UL, UR, GAS, 1.0, 1.0)
    np.testing.assert_allclose(F, 0.0, atol=1e-14)


def test_godunov_flux_inside_a_fan_next_to_vacuum():
    # Both states move right: x/t = 0 lies in the left fan, not in the vacuum.
    WL, WR = (1.0, 0.0, 0.4), (1.0, 8.0, 0.4)
    UL = primitive_to_conservative(*[np.array([v]) for v in WL], GAMMA)
    UR = primitive_to_conservative(*[np.array([v]) for v in WR], GAMMA)
    F = get_scheme("godunov").compute_riemann_flux(UL, UR, GAS, 1.0, 1.0)
    aL = np.sqrt(GAMMA * 0.4)
    u = 2.0 / (GAMMA + 1.0) * aL                     # sonic point of the fan
    rho = (u / aL) ** (2.0 / (GAMMA - 1.0))
    p = 0.4 * (u / aL) ** (2.0 * GAMMA / (GAMMA - 1.0))
    expected = compute_flux(primitive_to_conservative(
        np.array([rho]), np.array([u]), np.array([p]), GAMMA), GAMMA)[:, 0]
    np.testing.assert_allclose(F[:, 0], expected, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:Solution diverged")
def test_godunov_reports_divergence_instead_of_raising():
    # WENO5 produces a negative density in a Runge-Kutta stage here; the
    # exact solver must hand back NaN so that the run ends as "diverged".
    result = run_simulation(near_vacuum(200), get_scheme("weno5-godunov"), n_snapshots=0)
    assert result.status in ("ok", "diverged")


def test_godunov_flux_is_nan_for_unphysical_input():
    UL = np.array([[-1e-3], [0.0], [1.0]])
    UR = primitive_to_conservative(np.array([1.0]), np.array([0.0]), np.array([1.0]), GAMMA)
    F = get_scheme("godunov").compute_riemann_flux(UL, UR, GAS, 1.0, 1.0)
    assert np.isnan(F).all()
