"""Plotly figures of the eight screens of the walkthrough page (labels in French).

Each figure shows one curve or one diagram and one bubble: an arrow annotation
pointing at the thing to look at.  Screens 1 to 6 advect a square wave
(euler1d.advection), screens 7 and 8 come back to Sod.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from euler1d.advection import SQUARE, cell_centres, cut_cell, exact_averages, exact_point
from euler1d.config import GasProperties
from euler1d.fv_walkthrough import SOD_LEFT, SOD_RIGHT, exact_waves, model_waves
from euler1d.physics import primitive_to_conservative
from euler1d.riemann import sample_riemann, solve_riemann
from ui_common import COLORS, THESIS_LAYOUT, thesis_axis

CFL_SCREEN_4 = 0.8
CFL_SCREEN_6 = 0.5
GAMMA = 1.4
GREY = "#9a9a9a"
BLUE = COLORS[0]
RED = "#d9534f"
Y_RANGE = [-0.5, 1.5]
FINE = np.linspace(0.0, 1.0, 2001)

SLOPE_LABELS = {"order1": "ordre 1", "fromm": "pente sans limiteur",
                "minmod": "pente limitée minmod", "van-leer": "pente limitée van Leer"}


def _fr(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def bubble(fig: go.Figure, x: float, y: float, text: str, ax: float = 40, ay: float = -50,
           xanchor: str = "auto") -> None:
    """The one thing to look at: an arrow from a framed label to (x, y)."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, ax=ax, ay=ay, xanchor=xanchor,
                       arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=RED,
                       bgcolor="#fff4f3", bordercolor=RED, borderwidth=1.5, borderpad=6,
                       font=dict(size=15, color="#222222"))


def _plateaus(values, name, color=BLUE, width=3.0, dash="solid"):
    """Cell averages drawn as horizontal plateaus over each cell of [0, 1]."""
    n = len(values)
    xs, ys = [], []
    for i, v in enumerate(values):
        xs += [i / n, (i + 1) / n, None]
        ys += [v, v, None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name, line=dict(color=color, width=width, dash=dash))


def _exact_curve(t: float, name: str = "solution exacte", color=GREY, dash="dash"):
    return go.Scatter(x=FINE, y=exact_point(FINE, t), mode="lines", name=name,
                      line=dict(color=color, width=2, dash=dash))


def _layout(fig: go.Figure, x_title="x", y_title="u", y_range=Y_RANGE, x_range=(0.0, 1.0),
            height=400) -> go.Figure:
    fig.update_layout(**THESIS_LAYOUT, height=height,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_layout(margin=dict(l=60, r=20, t=50, b=50))
    fig.update_xaxes(thesis_axis(title_text=x_title), range=list(x_range))
    fig.update_yaxes(thesis_axis(title_text=y_title), range=list(y_range))
    return fig


def _clip(y):
    """Values off the frame are drawn on its border."""
    return np.clip(y, Y_RANGE[0] + 0.05, Y_RANGE[1] - 0.05)


def _inner_bubble(fig, x, y, text, ay=-60, dx=40):
    """Bubble on [0, 1] whose label stays inside the frame: it opens toward the centre."""
    if x < 0.5:
        bubble(fig, x, y, text, ax=dx, ay=ay, xanchor="left")
    else:
        bubble(fig, x, y, text, ax=-dx, ay=ay, xanchor="right")


def _square_middle(t: float) -> float:
    """Middle of the longest visible piece of the square at time t."""
    a = (SQUARE[0] + t) % 1.0
    b = a + SQUARE[1] - SQUARE[0]
    if b <= 1.0:
        return 0.5 * (a + b)
    return 0.5 * (a + 1.0) if 1.0 - a >= b - 1.0 else 0.5 * (b - 1.0)


def _front(t: float) -> float:
    """Leading (right) edge of the square at time t."""
    return (SQUARE[1] + t) % 1.0


def _time(n_cells: int, cfl: float, steps: int) -> float:
    return steps * cfl / n_cells


# ---------------------------------------------------------------------------
# Chapter 1: advection of a square wave
# ---------------------------------------------------------------------------

def fig_screen_1(t: float) -> go.Figure:
    """The exact solution at time t, over its initial position."""
    fig = go.Figure([_exact_curve(0.0, "t = 0", color="#d0d0d0", dash="dot"),
                     _exact_curve(t, f"t = {_fr(t)}", color="black", dash="solid")])
    fig.data[1].line.width = 3
    _inner_bubble(fig, _square_middle(t), 1.0, "il avance sans se déformer")
    return _layout(fig)


def fig_screen_2(n_cells: int) -> go.Figure:
    """Exact profile and its cell averages at t = 0; the bubble on a cut cell."""
    u = exact_averages(n_cells, 0.0)
    fig = go.Figure([_exact_curve(0.0), _plateaus(u, f"moyennes sur {n_cells} mailles")])
    cut = cut_cell(n_cells)
    if cut < 0:
        cut = next((i for i, v in enumerate(u) if 1e-9 < v < 1.0 - 1e-9), -1)
    if cut >= 0:
        fig.add_vrect(x0=cut / n_cells, x1=(cut + 1) / n_cells, fillcolor=RED, opacity=0.15,
                      line_width=0)
        _inner_bubble(fig, (cut + 0.5) / n_cells, u[cut], "valeur intermédiaire : la moyenne",
                      ay=-70)
    else:
        _inner_bubble(fig, SQUARE[0], 0.5, "bord sur une face : aucune maille coupée", ay=-80)
    return _layout(fig)


def _flux_arrow(fig, x_face, y, length, label, color, side):
    """Horizontal arrow through a face (a = 1: flux goes to the right)."""
    fig.add_trace(go.Scatter(x=[x_face - 0.5 * length, x_face + 0.5 * length], y=[y, y],
                             mode="lines+markers", showlegend=False, hoverinfo="skip",
                             line=dict(color=color, width=3),
                             marker=dict(symbol=["line-ns", "triangle-right"], size=[1, 14],
                                         color=color)))
    fig.add_annotation(x=x_face, y=y, text=label, showarrow=False, yshift=18,
                       xanchor=side, xshift=-6 if side == "right" else 6,
                       font=dict(size=14, color=color))


def fig_screen_3(u_before: np.ndarray, u_after: np.ndarray, cell: int, cfl: float) -> go.Figure:
    """One time step seen from one cell: what enters, what leaves, before and after."""
    n = len(u_before)
    dx = 1.0 / n
    left, right = cell * dx, (cell + 1) * dx
    # Amounts moved during the step: nu * a u* (upwind, a = 1, nu = a dt / dx).
    f_in, f_out = cfl * u_before[cell - 1], cfl * u_before[cell]
    fig = go.Figure([_plateaus(u_before, "avant le pas", color=GREY, width=2, dash="dash"),
                     _plateaus(u_after, "après le pas")])
    fig.add_vrect(x0=left, x1=right, fillcolor=RED, opacity=0.12, line_width=0)
    y_arrow = 1.3
    _flux_arrow(fig, left, y_arrow, 1.2 * dx, f"entre {_fr(f_in)}", COLORS[2], "right")
    _flux_arrow(fig, right, y_arrow, 1.2 * dx, f"sort {_fr(f_out)}", COLORS[3], "left")
    bubble(fig, 0.5 * (left + right), u_after[cell],
           f"{_fr(u_before[cell])} + {_fr(f_in)} − {_fr(f_out)} = {_fr(u_after[cell])}",
           ax=110, ay=40)
    centre = 0.5 * (left + right)
    return _layout(fig, x_range=(centre - 6.5 * dx, centre + 6.5 * dx))


def _explosion_bubble(fig, run) -> None:
    """Points at the largest value of the last state, drawn on the frame border if off it."""
    n = len(run.u)
    i = int(np.argmax(np.abs(run.u)))
    y = float(_clip(run.u[i]))
    _inner_bubble(fig, (i + 0.5) / n, y, f"le calcul explose au pas {run.steps_done + 1}",
                  ay=60 if y > 0.5 else -60)


def fig_screen_4(run, scheme: str, n_steps: int) -> go.Figure:
    """Upwind or centred face value, after n_steps steps at CFL 0.8."""
    n = len(run.u)
    t = _time(n, CFL_SCREEN_4, run.steps_done)
    name = "décentré amont" if scheme == "upwind" else "centré"
    fig = go.Figure([_exact_curve(t), _plateaus(_clip(run.u), name)])
    if run.exploded:
        _explosion_bubble(fig, run)
    elif n_steps == 0:
        _inner_bubble(fig, _front(0.0), 0.5, "état initial : augmentez le nombre de pas")
    elif scheme == "upwind":
        _inner_bubble(fig, _front(t), 0.5, "le front s'étale : diffusion numérique")
    else:
        i = int(np.argmax(np.abs(run.u - 0.5)))
        _inner_bubble(fig, (i + 0.5) / n, _clip(run.u[i]),
                      "des oscillations apparaissent et grandissent", ay=-50 if run.u[i] < 1.0 else 50)
    return _layout(fig)


def fig_screen_5(run, cfl: float, n_steps: int) -> go.Figure:
    """Upwind scheme at the chosen CFL number."""
    n = len(run.u)
    t = _time(n, cfl, run.steps_done)
    fig = go.Figure([_exact_curve(t), _plateaus(_clip(run.u), f"décentré amont, CFL = {_fr(cfl)}")])
    if run.exploded:
        _explosion_bubble(fig, run)
    elif n_steps == 0:
        _inner_bubble(fig, _front(0.0), 0.5, "état initial : augmentez le nombre de pas")
    elif np.isclose(cfl, 1.0):
        _inner_bubble(fig, _square_middle(t), 1.0, "exact : un pas = une maille")
    elif cfl > 1.0:
        i = int(np.argmax(np.abs(run.u - 0.5)))
        _inner_bubble(fig, (i + 0.5) / n, _clip(run.u[i]), "CFL > 1 : le calcul va exploser",
                      ay=-50 if run.u[i] < 1.0 else 50)
    else:
        _inner_bubble(fig, _front(t), 0.5, "étalé : CFL < 1")
    return _layout(fig)


def fig_screen_6(u: np.ndarray, slope: str, t: float) -> go.Figure:
    """Order 1 against a slope in each cell, with or without a limiter, at time t."""
    n = len(u)
    fig = go.Figure([_exact_curve(t), _plateaus(u, SLOPE_LABELS[slope])])
    if t == 0.0:
        _inner_bubble(fig, _front(0.0), 0.5, "état initial : faites avancer t")
    elif slope == "fromm":
        i = int(np.argmax(u))
        _inner_bubble(fig, (i + 0.5) / n, _clip(u[i]), "dépassements", ay=-50)
    elif slope == "order1":
        _inner_bubble(fig, _front(t), 0.5, "très étalé")
    else:
        _inner_bubble(fig, _front(t), 0.5, "front plus raide, sans oscillation")
    return _layout(fig)


# ---------------------------------------------------------------------------
# Chapter 2: Euler (Sod)
# ---------------------------------------------------------------------------

_WAVE_NAMES = {"rarefaction": "détente", "contact": "contact", "shock": "choc"}
_FLUX_NAMES = {"rusanov": "Rusanov", "hll": "HLL", "hllc": "HLLC"}


def fig_screen_7(choice: str) -> go.Figure:
    """(x, t) diagram at the central Sod face: exact waves, and the waves a solver models."""
    waves = exact_waves(SOD_LEFT, SOD_RIGHT, GAMMA)
    fig = go.Figure()

    def ray(speed, name, show, **line):
        fig.add_trace(go.Scatter(x=[0.0, speed], y=[0.0, 1.0], mode="lines", line=line, name=name,
                                 showlegend=show, legendgroup=name, hoverinfo="skip"))

    black = dict(color="black")
    for wave in waves:
        if wave.kind == "rarefaction":
            for j, s in enumerate(np.linspace(*wave.speeds, 6)):
                ray(s, "solution exacte", False, width=1, **black)
        elif wave.kind == "contact":
            ray(wave.speeds[0], "solution exacte", False, width=2.5, dash="dash", **black)
        else:
            ray(wave.speeds[0], "solution exacte", True, width=3.5, **black)
        tip = wave.speeds[0] if wave.kind != "rarefaction" else np.mean(wave.speeds)
        fig.add_annotation(x=tip, y=1.0, text=_WAVE_NAMES[wave.kind], showarrow=False, yshift=12,
                           font=dict(size=14))
    contact = waves[1].speeds[0]
    if choice == "exact":
        bubble(fig, -0.6, 0.6, "trois ondes, dans les deux sens", ax=-110, ay=130)
    else:
        UL = primitive_to_conservative(*(np.array([v]) for v in SOD_LEFT), GAMMA)
        UR = primitive_to_conservative(*(np.array([v]) for v in SOD_RIGHT), GAMMA)
        speeds = model_waves(choice, np.ravel(UL), np.ravel(UR), GasProperties(gamma=GAMMA))
        for j, s in enumerate(speeds):
            ray(s, f"ondes du modèle {_FLUX_NAMES[choice]}", j == 0, width=3, dash="dot",
                color=RED)
        if choice == "hll":
            bubble(fig, 0.6 * contact, 0.6, "HLL ne voit pas cette onde :<br>le contact sera étalé",
                   ax=135, ay=130)
        else:
            bubble(fig, 0.6 * contact, 0.6, "les trois ondes sont modélisées", ax=135, ay=130)
    fig.add_vline(x=0.0, line=dict(color=COLORS[2], width=2))
    fig.add_annotation(x=0.0, y=0.03, text="face", showarrow=False, xshift=-22,
                       font=dict(size=14, color=COLORS[2]))
    return _layout(fig, x_title="x − x_face (pour t = 1)", y_title="t", y_range=(0.0, 1.12),
                   x_range=(-2.0, 2.0), height=430)


def fig_screen_8(x: np.ndarray, rho: np.ndarray, flux: str, k: int) -> go.Figure:
    """Density of Sod at t = 0.2 on 100 cells; the bubble on the contact."""
    t = 0.2
    rho_exact, _, _ = sample_riemann(FINE, t, *SOD_LEFT, *SOD_RIGHT, GAMMA, x0=0.5)
    _, u_star, rho_l, rho_r = solve_riemann(*SOD_LEFT, *SOD_RIGHT, GAMMA)
    fig = go.Figure([go.Scatter(x=FINE, y=rho_exact, mode="lines", name="solution exacte",
                                line=dict(color=GREY, width=2, dash="dash")),
                     _plateaus(rho, _FLUX_NAMES[flux])])
    word = "maille" if k <= 1 else "mailles"
    _inner_bubble(fig, 0.5 + u_star * t, 0.5 * (rho_l + rho_r), f"contact étalé sur {k} {word}",
                  ay=-90)
    return _layout(fig, y_title="ρ", y_range=(0.0, 1.1))
