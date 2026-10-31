"""Streamlit app: navigation and the finite-volume walkthrough page (AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ENTRY = str(Path(__file__).resolve().parents[1] / "Solveur_Euler_1D.py")


def _app():
    at = AppTest.from_file(ENTRY, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home_is_the_finite_volume_page():
    at = _app()
    assert any("volumes finis" in t.value for t in at.title)


def test_solver_page_runs():
    at = _app()
    at.switch_page("pages/solver.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(s.label == "Test case" for s in at.selectbox)


def test_fourier_page_runs():
    at = _app()
    at.switch_page("pages/2_Fourier_Analysis.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Fourier" in t.value for t in at.title)
