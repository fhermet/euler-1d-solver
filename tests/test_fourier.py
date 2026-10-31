"""Fourier analysis against amplification factors known in closed form."""

import numpy as np
import pytest

from euler1d.fourier_analysis import compute_amplification
from euler1d.schemes import get_scheme

GAMMA = 1.4
C0 = np.sqrt(GAMMA)
U0 = 1.0
SPEEDS = {"entropy": U0, "acoustic": U0 + C0, "acoustic-left": U0 - C0}


def _analytic(name, theta, cfl, speed):
    """G of the linearised scheme with forward Euler, dt = cfl dx / (|u0| + c0)."""
    dtdx = cfl / (U0 + C0)
    nu = speed * dtdx
    if name == "lax-friedrichs":
        return np.cos(theta) - 1j * nu * np.sin(theta)
    if name == "lax-wendroff":
        return 1 - 1j * nu * np.sin(theta) - nu**2 * (1 - np.cos(theta))
    # Rusanov: dissipation at the largest speed |u0| + c0 for every wave.
    return 1 - cfl * (1 - np.cos(theta)) - 1j * nu * np.sin(theta)


def _phase_ratio(G, theta, cfl, speed):
    return np.unwrap(np.angle(G)) / (-speed * cfl / (U0 + C0) * theta)


@pytest.mark.parametrize("wave", list(SPEEDS))
@pytest.mark.parametrize("name", ["lax-friedrichs", "lax-wendroff", "rusanov"])
def test_matches_closed_form(name, wave):
    r = compute_amplification(get_scheme(name), cfl=0.5, n_cells=64, n_modes=32, wave_type=wave)
    G = _analytic(name, r["theta"], 0.5, SPEEDS[wave])
    np.testing.assert_allclose(r["abs_G"], np.abs(G), atol=1e-6)
    ok = np.isfinite(r["phase_ratio"])
    # The phase error grows like (noise on G ~ 1e-7) / |G| for damped modes.
    np.testing.assert_allclose(r["phase_ratio"][ok],
                               _phase_ratio(G, r["theta"], 0.5, SPEEDS[wave])[ok], rtol=1e-4)


def test_phase_is_undefined_at_the_nyquist_mode():
    # theta = pi is a standing sawtooth on the grid: no phase.
    r = compute_amplification(get_scheme("rusanov"), n_cells=64, n_modes=32)
    assert r["theta"][-1] == pytest.approx(np.pi)
    assert np.isnan(r["phase_ratio"][-1])


def test_requested_time_integrator_is_used():
    # Heun on the Rusanov operator: G = 1 + z + z^2/2 with z = G_Euler - 1.
    r = compute_amplification(get_scheme("rusanov"), cfl=0.5, n_cells=64, n_modes=32,
                              time_integrator="RK2")
    z = _analytic("rusanov", r["theta"], 0.5, U0) - 1
    np.testing.assert_allclose(r["abs_G"], np.abs(1 + z + z**2 / 2), atol=1e-6)
    assert r["time_integrator"] == "RK2"


def test_cfl_follows_the_scheme_limit_unless_asked_otherwise():
    scheme = get_scheme("eno2-hllc")
    assert compute_amplification(scheme, cfl=0.9, n_cells=32, n_modes=16)["cfl"] == 0.5
    r = compute_amplification(scheme, cfl=0.9, n_cells=128, n_modes=64,
                              wave_type="acoustic", respect_cfl_limit=False)
    assert r["cfl"] == 0.9
    # ENO2 + RK2 is linearly unstable at CFL 0.9 near theta = pi.
    assert r["abs_G"].max() > 1.4
