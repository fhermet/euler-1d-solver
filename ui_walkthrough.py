"""Plotly figures of the finite-volume walkthrough page (labels in French)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from euler1d.fv_walkthrough import FluxBreakdown, Wave
from euler1d.physics import conservative_to_primitive
from ui_common import COLORS, THESIS_LAYOUT, thesis_axis

VAR_NAMES = ["ρ", "u", "p"]
COMPONENT_NAMES = ["masse", "quantité de mouvement", "énergie"]
GREY = "#9a9a9a"


def _prim(U: np.ndarray, gamma: float) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array(conservative_to_primitive(U, gamma))


def _faces(x: np.ndarray, dx: float) -> np.ndarray:
    return np.append(x - 0.5 * dx, x[-1] + 0.5 * dx)


def _plateaus(x, dx, values, name, color, dash="solid", width=2.0, opacity=1.0):
    """Cell averages drawn as horizontal plateaus over each cell."""
    xs, ys = [], []
    for xc, v in zip(x, values):
        xs += [xc - 0.5 * dx, xc + 0.5 * dx, None]
        ys += [v, v, None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name, opacity=opacity,
                      line=dict(color=color, width=width, dash=dash))


def _finish(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(**THESIS_LAYOUT, height=height,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def _face_lines(fig, x, dx, row=None, col=None):
    for xf in _faces(x, dx):
        fig.add_vline(x=xf, line=dict(color="#e3e3e3", width=1), row=row, col=col)


def fig_averages(x, dx, U, var, gamma, exact) -> go.Figure:
    """Step 1: cell averages as plateaus, shaded area under each one."""
    W = _prim(U, gamma)
    custom = np.column_stack([U[0], U[1], U[2], W[0], W[1], W[2]])
    fig = go.Figure()
    _face_lines(fig, x, dx)
    fig.add_trace(go.Bar(
        x=x, y=W[var], width=dx, name="contenu de la maille",
        marker=dict(color=COLORS[0], opacity=0.18, line=dict(width=0)),
        customdata=custom,
        hovertemplate=("ρ = %{customdata[0]:.4f}, ρu = %{customdata[1]:.4f}, "
                       "E = %{customdata[2]:.4f}<br>ρ = %{customdata[3]:.4f}, "
                       "u = %{customdata[4]:.4f}, p = %{customdata[5]:.4f}<extra></extra>"),
    ))
    fig.add_trace(_plateaus(x, dx, W[var], "moyennes de maille", COLORS[0]))
    if exact is not None:
        fig.add_trace(_plateaus(x, dx, exact, "solution exacte", GREY, "dot", 1.2))
    fig.update_xaxes(thesis_axis(title_text="x"))
    fig.update_yaxes(thesis_axis(title_text=VAR_NAMES[var]))
    return _finish(fig)


def fig_reconstruction(x, dx, U, UL, UR, flat, var, gamma, linear, exact) -> go.Figure:
    """Step 2: reconstruction in each cell, states at the interfaces, jumps."""
    W = _prim(U, gamma)
    WL, WR = _prim(UL, gamma)[var], _prim(UR, gamma)[var]
    xf = _faces(x, dx)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08)
    _face_lines(fig, x, dx, row=1, col=1)
    fig.add_trace(_plateaus(x, dx, W[var], "moyennes", COLORS[0], opacity=0.3), row=1, col=1)
    # Cell i goes from its left face state UR[:, i] to its right face state UL[:, i+1].
    xs, ys = [], []
    for i in range(len(x)):
        xs += [xf[i], xf[i + 1], None]
        ys += [WR[i], WL[i + 1], None]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        name="reconstruction" if linear else "valeurs aux faces (reliées)",
        line=dict(color=COLORS[1], width=2.5, dash="solid" if linear else "dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=xf, y=WL, mode="markers", name="U_L (maille de gauche)",
                             marker=dict(symbol="triangle-left", size=9, color=COLORS[2])),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=xf, y=WR, mode="markers", name="U_R (maille de droite)",
                             marker=dict(symbol="triangle-right", size=9, color=COLORS[3])),
                  row=1, col=1)
    if np.any(flat):
        fig.add_trace(go.Scatter(x=x[flat], y=W[var][flat], mode="markers",
                                 name="pente annulée",
                                 marker=dict(symbol="x", size=11, color="black")),
                      row=1, col=1)
    if exact is not None:
        fig.add_trace(_plateaus(x, dx, exact, "solution exacte", GREY, "dot", 1.2),
                      row=1, col=1)
    fig.add_trace(go.Bar(x=xf, y=np.abs(WR - WL), width=0.35 * dx, name="|U_R − U_L|",
                         marker=dict(color=COLORS[4])), row=2, col=1)
    fig.update_yaxes(thesis_axis(title_text=VAR_NAMES[var]), row=1, col=1)
    fig.update_yaxes(thesis_axis(title_text="saut"), row=2, col=1)
    fig.update_xaxes(thesis_axis(title_text="x"), row=2, col=1)
    fig.update_xaxes(thesis_axis(), row=1, col=1)
    return _finish(fig, 560)


def fig_riemann(WL, WR, waves: list[Wave], model, dx, dt, var) -> go.Figure:
    """Step 3: the two states and the (x, t) diagram of the local problem."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65],
                        subplot_titles=["états de part et d'autre de l'interface",
                                        "diagramme (x, t) pendant un pas"])
    fig.add_trace(go.Scatter(x=[-1, 0, None, 0, 1], y=[WL[var], WL[var], None, WR[var], WR[var]],
                             mode="lines", name="U_L | U_R", line=dict(color=COLORS[0], width=3)),
                  row=1, col=1)
    fig.update_xaxes(thesis_axis(title_text="position (interface en 0)"), row=1, col=1)
    fig.update_yaxes(thesis_axis(title_text=VAR_NAMES[var]), row=1, col=1)

    def ray(speed, **line):
        return go.Scatter(x=[0, speed * dt], y=[0, dt], mode="lines", line=line,
                          showlegend=False, hoverinfo="skip")

    styles = {"shock": dict(color="black", width=3),
              "contact": dict(color="black", width=2, dash="dash"),
              "rarefaction": dict(color="black", width=1)}
    for wave in waves:
        if wave.kind == "rarefaction":
            for s in np.linspace(*wave.speeds, 6):
                fig.add_trace(ray(s, **styles["rarefaction"]), row=1, col=2)
        elif wave.kind == "vacuum":
            a, b = wave.speeds
            fig.add_trace(go.Scatter(x=[0, a * dt, b * dt, 0], y=[0, dt, dt, 0], fill="toself",
                                     mode="lines", line=dict(width=0), name="vide",
                                     fillcolor="rgba(0,0,0,0.12)"), row=1, col=2)
        else:
            fig.add_trace(ray(wave.speeds[0], **styles[wave.kind]), row=1, col=2)
    # One legend entry per wave kind.
    for kind, label in [("shock", "choc"), ("contact", "contact"), ("rarefaction", "détente")]:
        if any(w.kind == kind for w in waves):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=styles[kind],
                                     name=f"solution exacte : {label}"), row=1, col=2)
    for j, s in enumerate(model or []):
        fig.add_trace(go.Scatter(x=[0, s * dt], y=[0, dt], mode="lines",
                                 line=dict(color=COLORS[1], width=2.5, dash="dot"),
                                 name="ondes du modèle du flux", showlegend=j == 0),
                      row=1, col=2)
    fig.add_vline(x=0, line=dict(color=COLORS[2], width=2), row=1, col=2)
    fig.add_shape(type="rect", x0=-dx, x1=dx, y0=0, y1=dt, line=dict(color=COLORS[4], dash="dash"),
                  row=1, col=2)
    fig.update_xaxes(thesis_axis(title_text="x − x_interface"), range=[-1.3 * dx, 1.3 * dx],
                     row=1, col=2)
    fig.update_yaxes(thesis_axis(title_text="t"), range=[0, 1.05 * dt], row=1, col=2)
    return _finish(fig, 460)


def fig_fluxes(x, dx, U, breakdown: FluxBreakdown, component, var, gamma, k) -> go.Figure:
    """Step 4: flux arrows at the interfaces and the flux decomposition at k."""
    W = _prim(U, gamma)
    F = breakdown.numerical[component]
    xf = _faces(x, dx)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32],
                        subplot_titles=[f"flux de {COMPONENT_NAMES[component]} aux interfaces",
                                        f"à l'interface {k}"])
    fig.add_trace(_plateaus(x, dx, W[var], "moyennes", COLORS[0]), row=1, col=1)
    top = float(np.nanmax(W[var]))
    span = top - float(np.nanmin(W[var])) or 1.0
    y_arrow = top + 0.15 * span
    scale = 0.45 * dx / max(float(np.nanmax(np.abs(F))), 1e-300)
    for j, (xj, fj) in enumerate(zip(xf, F)):
        if abs(fj) * scale < 0.02 * dx:
            continue
        fig.add_annotation(x=xj + fj * scale, y=y_arrow, ax=xj - fj * scale, ay=y_arrow,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=2,
                           arrowcolor=COLORS[1] if j != k else COLORS[3], text="")
    fig.update_yaxes(thesis_axis(title_text=VAR_NAMES[var]),
                     range=[float(np.nanmin(W[var])) - 0.1 * span, top + 0.3 * span], row=1, col=1)
    fig.update_xaxes(thesis_axis(title_text="x"), row=1, col=1)
    labels = ["moyenne centrée", "dissipation", "flux du schéma", "Godunov exact"]
    values = [breakdown.centred[component, k], breakdown.dissipation[component, k],
              breakdown.numerical[component, k], breakdown.godunov[component, k]]
    fig.add_trace(go.Bar(x=labels, y=values, name="décomposition",
                         marker=dict(color=[COLORS[0], COLORS[4], COLORS[1], GREY])),
                  row=1, col=2)
    fig.update_yaxes(thesis_axis(title_text="flux"), row=1, col=2)
    return _finish(fig, 440)


def fig_update(x, dx, U, U_new, F, i, var, gamma) -> go.Figure:
    """Step 5: one cell before and after its update, and the whole profile."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65],
                        subplot_titles=[f"maille {i} : moyennes conservatives",
                                        "profil avant et après un étage"])
    names = ["ρ", "ρu", "E"]
    fig.add_trace(go.Bar(x=names, y=U[:, i], name="avant", marker=dict(color=COLORS[0])),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=U_new[:, i], name="après", marker=dict(color=COLORS[1])),
                  row=1, col=1)
    W, W_new = _prim(U, gamma), _prim(U_new, gamma)
    fig.add_vrect(x0=x[i] - 0.5 * dx, x1=x[i] + 0.5 * dx, fillcolor=COLORS[3], opacity=0.12,
                  line_width=0, row=1, col=2)
    fig.add_trace(_plateaus(x, dx, W[var], "avant", COLORS[0]), row=1, col=2)
    fig.add_trace(_plateaus(x, dx, W_new[var], "après", COLORS[1], "dash"), row=1, col=2)
    fig.update_xaxes(thesis_axis(title_text="x"), row=1, col=2)
    fig.update_yaxes(thesis_axis(title_text=VAR_NAMES[var]), row=1, col=2)
    fig.update_layout(barmode="group")
    return _finish(fig, 420)


def fig_stages(x, dx, stages: list[np.ndarray], var, gamma, i_max) -> go.Figure:
    """Step 6: state at each Runge-Kutta stage, from U^n to U^{n+1}."""
    fig = go.Figure()
    _face_lines(fig, x, dx)
    dashes = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
    for s, U in enumerate(stages):
        if s == 0:
            name = "Uⁿ (étage 1)"
        elif s == len(stages) - 1:
            name = "Uⁿ⁺¹"
        else:
            name = f"étage {s + 1}"
        last = s == len(stages) - 1
        fig.add_trace(_plateaus(x, dx, _prim(U, gamma)[var], name,
                                COLORS[1] if last else COLORS[s % len(COLORS)],
                                "solid" if last else dashes[s % len(dashes)],
                                3 if last else 1.6))
    fig.add_vrect(x0=x[i_max] - 0.5 * dx, x1=x[i_max] + 0.5 * dx, fillcolor=COLORS[4],
                  opacity=0.15, line_width=0, annotation_text="max |u| + c",
                  annotation_position="top left")
    fig.update_xaxes(thesis_axis(title_text="x"))
    fig.update_yaxes(thesis_axis(title_text=VAR_NAMES[var]))
    return _finish(fig)
