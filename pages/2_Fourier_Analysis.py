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
    "of each scheme (reconstruction + flux + time integrator) over one time step, "
    "on an eigenmode of the Euler system linearized around $(\\rho, u, p) = (1, 1, 1)$. "
    "$|G| < 1$ = dissipation, $|G| > 1$ = instability, "
    "$\\phi_{num}/\\phi_{exact} \\neq 1$ = dispersion error."
)

selected = scheme_selector_multi("fourier")

WAVE_LABELS = {
    "entropy": "Entropy (speed u)",
    "acoustic": "Acoustic, right-going (speed u + c)",
    "acoustic-left": "Acoustic, left-going (speed u - c, nearly sonic)",
}

col1, col2, col3 = st.columns(3)
with col1:
    fourier_cfl = st.slider("CFL", 0.1, 1.0, 0.5, step=0.05, key="fourier_cfl")
with col2:
    wave_type = st.selectbox(
        "Wave type",
        list(WAVE_LABELS),
        format_func=WAVE_LABELS.get,
        help=(
            "Eigenmode of the linearized system that is excited. With u = 1 and "
            "c = 1.18, the left-going acoustic wave is slow (u - c = -0.18)."
        ),
        key="fourier_wave",
    )
with col3:
    ignore_limit = st.checkbox(
        "Ignore the scheme's CFL limit",
        value=False,
        help=(
            "By default the CFL is capped as in the solver (0.5 for MUSCL, ENO, "
            "WENO). Tick to analyse the requested CFL and explore stability: "
            "for instance ENO2 with RK2 becomes unstable (|G| > 1) above 0.5."
        ),
        key="fourier_ignore_limit",
    )

if selected:
    wave_label = {"entropy": "entropy", "acoustic": "right acoustic",
                  "acoustic-left": "left acoustic"}[wave_type]
    used = []
    with st.spinner("Fourier analysis..."):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Dissipation |G(θ)|", "Dispersion φ/φ_exact"],
            horizontal_spacing=0.1,
        )

        for i, (sname, limiter, ti) in enumerate(selected):
            kwargs = {"limiter": limiter} if limiter else {}
            scheme = get_scheme(sname, **kwargs)
            result = compute_amplification(
                scheme, cfl=fourier_cfl, wave_type=wave_type,
                time_integrator=None if ti == "Auto" else ti,
                respect_cfl_limit=not ignore_limit,
            )
            theta = result["theta"]
            color = COLORS[i % len(COLORS)]
            mkr = MARKERS[i % len(MARKERS)]
            dname = f"{scheme.name} [{result['time_integrator']}, CFL {result['cfl']:g}]"
            used.append((dname, result))

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

        if any(r["cfl"] < fourier_cfl for _, r in used):
            st.caption(
                "Some schemes were analysed at CFL 0.5, their guaranteed stability "
                "limit (as in the solver). Tick 'Ignore the scheme's CFL limit' to "
                "analyse them at the requested CFL."
            )
        unstable = [name for name, r in used if r["abs_G"].max() > 1.0 + 1e-6]
        if unstable:
            st.warning(
                "Unstable (|G| > 1 for some θ): " + ", ".join(unstable)
                + ". These modes grow at every time step."
            )

        st.markdown(
            "**Reading guide:** An ideal scheme would have $|G| = 1$ (no dissipation) and "
            "$\\phi/\\phi_{exact} = 1$ (no dispersion error) for all $\\theta$. "
            "In practice, dissipative schemes (Lax-Friedrichs, Rusanov) damp "
            "high frequencies, while low-dissipation schemes (Lax-Wendroff, JST) "
            "introduce dispersion errors that cause oscillations near shocks.\n\n"
            "**Limits of a linear analysis:** the perturbation is tiny ($10^{-6}$), so "
            "WENO weights take their ideal values: the curves show WENO in smooth "
            "regions. $|G| \\leq 1$ is linear stability only; it does not guarantee "
            "the TVD property. MUSCL (minmod, van Leer) at CFL 0.9 has $|G| \\leq 1$ here, yet the solver "
            "caps it at 0.5 because above that round-off can grow near shocks. "
            "The phase is not defined at $\\theta = \\pi$ (standing grid mode)."
        )
