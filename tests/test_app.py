"""Streamlit app: navigation and the finite-volume walkthrough page (AppTest)."""

import json
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


def _no_exception(at):
    assert not at.exception, [e.value for e in at.exception]


def _goto(at, screen):
    at.button(key=f"fv_goto_{screen}").click().run()
    _no_exception(at)


def _bubble_x(at):
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    [x] = [a["x"] for a in spec["layout"]["annotations"] if a.get("showarrow")]
    return x


def test_eight_screens_without_error():
    at = _app()
    for _ in range(7):
        at.button(key="fv_next").click().run()
        _no_exception(at)
        assert len(at.get("plotly_chart")) == 1
    assert at.session_state["fv_screen"] == 8
    at.button(key="fv_prev").click().run()
    assert at.session_state["fv_screen"] == 7


CONTROLS = {
    1: [("slider", "fv_t1", 0.6)],
    2: [("slider", "fv_n2", 50), ("slider", "fv_n2", 80)],
    3: [("button", "fv_step3", None), ("button", "fv_reset3", None)],
    4: [("radio", "fv_scheme4", "centred"), ("slider", "fv_n4", 40), ("slider", "fv_n4", 0)],
    5: [("slider", "fv_cfl5", 1.0), ("slider", "fv_cfl5", 1.3), ("slider", "fv_n5", 40),
        ("slider", "fv_n5", 0)],
    6: [("radio", "fv_slope6", s) for s in ("order1", "fromm", "minmod", "van-leer")]
       + [("slider", "fv_t6", 1.0)],
    7: [("radio", "fv_choice7", c) for c in ("exact", "hll", "hllc")],
    8: [("radio", "fv_flux8", f) for f in ("rusanov", "hll", "hllc")],
}


@pytest.mark.parametrize("screen", range(1, 9))
def test_every_control(screen):
    at = _app()
    _goto(at, screen)
    for kind, key, value in CONTROLS[screen]:
        if kind == "button":
            at.button(key=key).click().run()
        else:
            getattr(at, kind)(key=key).set_value(value).run()
        _no_exception(at)
        assert len(at.get("plotly_chart")) == 1


def test_screen3_follows_the_same_cell():
    at = _app()
    _goto(at, 3)
    start = _bubble_x(at)
    for _ in range(30):
        at.button(key="fv_step3").click().run()
    _no_exception(at)
    assert at.session_state["fv_steps3"] == 30
    assert _bubble_x(at) == start
    at.button(key="fv_reset3").click().run()
    assert at.session_state["fv_steps3"] == 0


def test_settings_survive_other_screen_and_page():
    at = _app()
    _goto(at, 4)
    at.radio(key="fv_scheme4").set_value("centred").run()
    _goto(at, 6)
    at.switch_page("pages/solver.py").run()
    at.switch_page("pages/finite_volumes.py").run()
    _no_exception(at)
    assert at.session_state["fv_screen"] == 6
    _goto(at, 4)
    assert at.radio(key="fv_scheme4").value == "centred"


def test_no_streamlit_warning():
    at = _app()
    for screen in range(1, 9):
        _goto(at, screen)
        assert not at.warning, [w.value for w in at.warning]


def test_screen3_hint_says_the_cell_empties():
    at = _app()
    _goto(at, 3)
    assert any("se vide" in i.value for i in at.info)


def test_screen5_hint_leads_to_an_explosion():
    at = _app()
    _goto(at, 5)
    assert any("40 pas" in i.value for i in at.info)
    at.slider(key="fv_cfl5").set_value(1.1).run()
    at.slider(key="fv_n5").set_value(40).run()
    assert "explose" in _bubble_text(at)


def _bubble_text(at):
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    [text] = [a["text"] for a in spec["layout"]["annotations"] if a.get("showarrow")]
    return text
