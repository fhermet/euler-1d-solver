"""Dissipation and dispersion analysis (Fourier)."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import numpy as np

from euler1d.fourier_analysis import compute_amplification, damping_per_wavelength
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

per_wavelength = st.radio(
    "Dissipation shown",
    ["per time step", "after travelling one wavelength"],
    horizontal=True,
    help=(
        "|G| is a factor per time step: its value depends on the time step. "
        "To compare schemes run at different CFL numbers, or one scheme at "
        "several CFL numbers, look at the amplitude left after the wave has "
        "travelled one wavelength, |G|^(2π/(θσ)), with σ the Courant number "
        "of the wave."
    ),
    key="fourier_norm",
) == "after travelling one wavelength"

if selected:
    wave_label = {"entropy": "entropy", "acoustic": "right acoustic",
                  "acoustic-left": "left acoustic"}[wave_type]
    used = []
    with st.spinner("Fourier analysis..."):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Amplitude after one wavelength" if per_wavelength else "Dissipation |G(θ)| per step",
                "Dispersion φ/φ_exact",
            ],
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
            dname = (
                f"{scheme.name} [{result['time_integrator']}, CFL {result['cfl']:g}, "
                f"σ {result['sigma']:.2f}]"
            )
            used.append((dname, result))
            damping = (
                damping_per_wavelength(theta, result["abs_G"], result["sigma"])
                if per_wavelength else result["abs_G"]
            )

            fig.add_trace(go.Scatter(
                x=theta, y=damping, mode="lines+markers",
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

        # θ = 2π / (cells per wavelength): label the ticks with both.
        ticks = [np.pi / 8, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
        ticktext = ["π/8<br>16 cells", "π/4<br>8 cells", "π/2<br>4 cells",
                    "3π/4<br>2.7 cells", "π<br>2 cells"]
        for col in (1, 2):
            fig.update_xaxes(
                thesis_axis(title_text="θ = k Δx  (cells per wavelength = 2π/θ)"),
                tickvals=ticks, ticktext=ticktext, row=1, col=col,
            )
        fig.update_yaxes(
            thesis_axis(title_text="amplitude after one wavelength" if per_wavelength else "|G|"),
            row=1, col=1,
        )
        # Lax-Friedrichs reaches phase ratios of 20 and more: bound the axis.
        fig.update_yaxes(thesis_axis(title_text="φ / φ_exact"), range=[0, 3], row=1, col=2)

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
        clipped = [name for name, r in used if np.nanmax(r["phase_ratio"]) > 3]
        if clipped:
            st.caption(
                "Phase ratio above 3 not shown for: " + ", ".join(clipped) + ". "
                "Where |G| is small the phase hardly matters: the mode is gone."
            )
        unstable = [name for name, r in used if r["abs_G"].max() > 1.0 + 1e-6]
        if unstable:
            st.warning(
                "|G| > 1 for some θ: " + ", ".join(unstable) + ". For a linear "
                "scheme (first order, Lax-Wendroff, JST, WENO in smooth regions) "
                "these modes grow at every step: the scheme is unstable. For a "
                "nonlinear one (MUSCL limiters, ENO) G describes a single step: "
                "the growth is transient, as the stencil choice changes with the "
                "waveform."
            )

        st.markdown(
            "**Reading guide:** An ideal scheme would have $|G| = 1$ (no dissipation) and "
            "$\\phi/\\phi_{exact} = 1$ (no dispersion error) for all $\\theta$. "
            "Rusanov and the other upwind fluxes damp high frequencies; "
            "low-dissipation schemes (Lax-Wendroff) introduce dispersion errors "
            "that cause oscillations near shocks. Lax-Friedrichs is an exception: "
            "it damps intermediate frequencies strongly but not $\\theta = \\pi$, "
            "where $|G| = 1$: the odd and even cells are decoupled, and a "
            "sawtooth survives.\n\n"
            "**Why some curves coincide:** linearized around a uniform state, "
            "HLLC and Roe both upwind each wave at its own speed, so they give the "
            "same G on every wave. Rusanov damps every wave at the largest speed "
            "|u| + c: it coincides with them on the right-going acoustic wave, "
            "which travels at that speed, and is more dissipative on the slower "
            "entropy and left-going waves. HLLC and Roe differ only through "
            "nonlinear effects, at finite jumps (Sod, stationary contact).\n\n"
            "**Limits of a linear analysis:** the perturbation is tiny ($10^{-6}$), so "
            "WENO weights take their ideal values and the JST shock sensor is negligible "
            ": the curves show WENO and JST in smooth regions, not near shocks. "
            "$|G| \\leq 1$ is linear stability only; it does not guarantee "
            "the TVD property. MUSCL (minmod, van Leer) at CFL 0.9 has $|G| \\leq 1$ here, yet the solver "
            "caps it at 0.5 because above that round-off can grow near shocks. "
            "The phase is not defined at $\\theta = \\pi$ (standing grid mode)."
        )
