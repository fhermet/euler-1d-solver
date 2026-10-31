"""Choice of reconstructed variables: primitive, conservative, characteristic."""

import numpy as np
import pytest

from euler1d.boundary import apply_boundary_conditions
from euler1d.config import GasProperties
from euler1d.physics import (
    compute_flux, conservative_to_primitive, primitive_to_conservative,
)
from euler1d.schemes import get_scheme
from euler1d.schemes.reconstruction.variables import roe_eigenvectors

GAMMA = 1.4


def test_left_and_right_eigenvectors_are_inverse():
    rng = np.random.default_rng(0)
    n = 7
    UL = primitive_to_conservative(rng.uniform(0.5, 2, n), rng.uniform(-2, 2, n),
                                   rng.uniform(0.5, 2, n), GAMMA)
    UR = primitive_to_conservative(rng.uniform(0.5, 2, n), rng.uniform(-2, 2, n),
                                   rng.uniform(0.5, 2, n), GAMMA)
    L, R = roe_eigenvectors(UL, UR, GAMMA)
    product = np.einsum("ijk,jlk->ilk", L, R)
    np.testing.assert_allclose(product, np.eye(3)[:, :, None] * np.ones(n), atol=1e-12)


def _truncation_order(name, variables, Ns=(40, 80, 160)):
    """Order of the spatial operator on a smooth state where rho, u, p all vary."""
    scheme = get_scheme(name, variables=variables)
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(8)

    def U_exact(x):
        return primitive_to_conservative(1 + 0.2 * np.sin(2 * np.pi * x),
                                         1 + 0.3 * np.sin(2 * np.pi * x + 1),
                                         1 + 0.2 * np.cos(2 * np.pi * x), GAMMA)

    errors = []
    for N in Ns:
        dx = 1.0 / N
        xf = np.linspace(0.0, 1.0, N + 1)
        xc = 0.5 * (xf[:-1] + xf[1:])
        U_bar = sum(w * U_exact(xc + 0.5 * dx * q) for q, w in zip(gauss_x, gauss_w)) / 2
        U_ext = apply_boundary_conditions(U_bar, "periodic", scheme.n_ghost)
        F = scheme.compute_fluxes(U_ext, GasProperties(GAMMA), dx, 1e-3)
        F_exact = compute_flux(U_exact(xf), GAMMA)
        errors.append(np.abs((F[:, 1:] - F[:, :-1]) - (F_exact[:, 1:] - F_exact[:, :-1])).max() / dx)
    return np.log2(errors[-2] / errors[-1])


def test_weno5_on_primitive_variables_is_second_order_only():
    # W(U_bar) differs from the cell average of W by O(dx^2).
    assert _truncation_order("wenoz5-hllc", "primitive") < 2.5


@pytest.mark.parametrize("variables", ["conservative", "characteristic"])
def test_weno5_recovers_fifth_order(variables):
    assert _truncation_order("wenoz5-hllc", variables) > 4.5


def test_unknown_variables_raise():
    with pytest.raises(ValueError, match="variables"):
        get_scheme("weno5-hllc", variables="entropy")


@pytest.mark.parametrize("variables", ["primitive", "conservative", "characteristic"])
@pytest.mark.parametrize("recon", ["muscl", "eno2", "weno3", "weno5"])
def test_face_states_stay_physical_next_to_near_vacuum(recon, variables):
    # A steep drop to near vacuum makes high-order faces overshoot below zero;
    # such faces must fall back to the cell value.
    n = 16
    rho = np.where(np.arange(n) < 8, 1.0, 1e-6)
    p = np.where(np.arange(n) < 8, 1.0, 1e-8)
    U = primitive_to_conservative(rho, np.zeros(n), p, GAMMA)
    scheme = get_scheme(f"{recon}-hll", variables=variables)
    UL, UR = scheme.reconstruct(U, GAMMA, 1.0 / n)
    for face in (UL, UR):
        r, _, pf = conservative_to_primitive(face, GAMMA)
        assert (r > 0).all() and (pf > 0).all()
