"""Dissipation and dispersion analysis (Fourier)."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from euler1d.fourier_analysis import compute_amplification
from euler1d.schemes import get_scheme
from ui_common import (
    COLORS,
    MARKERS,
    THESIS_LAYOUT,
    _LEGEND_COMMON,
    scheme_selector_multi,
    thesis_axis,
)

st.set_page_config(page_title="Fourier Analysis", layout="wide")
st.title("Fourier Analysis")

st.markdown(
    "Linearized Fourier analysis: we measure the amplification factor $G(\\theta)$ "
    "of each scheme on an eigenmode of the linearized Euler system. "
    "$|G| < 1$ = dissipation, $\\phi_{num}/\\phi_{exact} \\neq 1$ = dispersion error."
)

selected = scheme_selector_multi("fourier")

col1, col2 = st.columns(2)
with col1:
    fourier_cfl = st.slider("CFL", 0.1, 0.9, 0.5, step=0.05, key="fourier_cfl")
with col2:
    wave_type = st.selectbox(
        "Wave type",
        ["entropie", "acoustique"],
        format_func=lambda x: {"entropie": "Entropy", "acoustique": "Acoustic"}[x],
        help="Entropy: pure advection (velocity u). Acoustic: pressure wave (velocity u+c).",
        key="fourier_wave",
    )

if selected:
    wave_label = {"entropie": "entropy", "acoustique": "acoustic"}[wave_type]
    with st.spinner("Fourier analysis..."):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Dissipation |G(θ)|", "Dispersion φ/φ_exact"],
            horizontal_spacing=0.1,
        )

        for i, (sname, limiter, _ti) in enumerate(selected):
            kwargs = {"limiter": limiter} if limiter else {}
            scheme = get_scheme(sname, **kwargs)
            result = compute_amplification(scheme, cfl=fourier_cfl, wave_type=wave_type)
            theta = result["theta"]
            color = COLORS[i % len(COLORS)]
            mkr = MARKERS[i % len(MARKERS)]
            dname = scheme.name

            fig.add_trace(go.Scatter(
                x=theta, y=result["abs_G"], mode="lines+markers",
                name=dname, legendgroup=dname,
                line=dict(color=color, width=2),
                marker=dict(symbol=mkr, size=5, maxdisplayed=25),
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=theta, y=result["phase_ratio"], mode="lines+markers",
                name=dname, legendgroup=dname, showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(symbol=mkr, size=5, maxdisplayed=25),
            ), row=1, col=2)

        fig.add_hline(y=1.0, line=dict(color="black", width=1, dash="dash"), row=1, col=1)
        fig.add_hline(y=1.0, line=dict(color="black", width=1, dash="dash"), row=1, col=2)

        fig.update_xaxes(thesis_axis(title_text="θ = k Δx"), row=1, col=1)
        fig.update_xaxes(thesis_axis(title_text="θ = k Δx"), row=1, col=2)
        fig.update_yaxes(thesis_axis(title_text="|G|"), row=1, col=1)
        fig.update_yaxes(thesis_axis(title_text="φ / φ_exact"), row=1, col=2)

        fig.update_layout(
            **THESIS_LAYOUT,
            title=dict(
                text=f"Fourier analysis — {wave_label} wave — CFL = {fourier_cfl}",
                font=dict(family="STIX Two Text, serif", size=17),
                x=0.0, xanchor="left",
            ),
            height=500, width=1200,
            legend=dict(**_LEGEND_COMMON),
        )
        fig.update_annotations(font=dict(family="STIX Two Text, serif", size=15), yshift=15)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Reading guide:** An ideal scheme would have $|G| = 1$ (no dissipation) and "
            "$\\phi/\\phi_{exact} = 1$ (no dispersion error) for all $\\theta$. "
            "In practice, dissipative schemes (Lax-Friedrichs, Rusanov) damp "
            "high frequencies, while low-dissipation schemes (Lax-Wendroff, JST) "
            "introduce dispersion errors that cause oscillations near shocks."
        )
