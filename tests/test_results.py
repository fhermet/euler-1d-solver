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
