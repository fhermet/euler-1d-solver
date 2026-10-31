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


def test_walk_through_six_steps():
    at = _app()
    for _ in range(5):
        at.button(key="fv_next").click().run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["fv_step"] == 6
    at.button(key="fv_goto_2").click().run()
    assert at.session_state["fv_step"] == 2


def test_setting_change_resets_time_keeps_step():
    at = _app()
    at.button(key="fv_goto_4").click().run()
    at.button(key="fv_step_btn").click().run()
    assert at.session_state["fv_t"] > 0
    at.selectbox(key="fv_flux").select("rusanov").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["fv_t"] == 0
    assert at.session_state["fv_step"] == 4


def test_ten_steps_then_reset():
    at = _app()
    at.button(key="fv_ten_btn").click().run()
    assert at.session_state["fv_n"] == 10 and at.session_state["fv_t"] > 0
    at.button(key="fv_reset_btn").click().run()
    assert at.session_state["fv_n"] == 0


def test_divergence_disables_time_buttons():
    # AUSM+ at first order diverges on the Sod start-up at CFL 0.9.
    at = _app()
    at.selectbox(key="fv_flux").select("ausm+").run()
    at.selectbox(key="fv_recon").select("constant").run()
    at.button(key="fv_ten_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.error, "a divergence message is shown"
    assert at.session_state["fv_n"] == 0
    assert at.button(key="fv_step_btn").disabled


def test_selection_clamped_when_mesh_shrinks():
    at = _app()
    at.slider(key="fv_cells").set_value(32).run()
    at.button(key="fv_goto_3").click().run()
    at.slider(key="fv_interface").set_value(32).run()
    at.slider(key="fv_cells").set_value(8).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["fv_interface"] <= 8
