"""Shared UI constants, styles, and scheme selector widgets for Streamlit pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from euler1d.schemes import get_scheme
from euler1d.schemes.reconstruction import available_limiters
from euler1d.solver import select_time_integrator

# ---------------------------------------------------------------------------
# Scheme selection constants
# ---------------------------------------------------------------------------

ALL_FLUX = ["lax-friedrichs", "rusanov", "hll", "hllc", "roe", "roe-nc", "godunov", "ausm+", "ausm+-up", "lax-wendroff", "jst"]
FLUX_LABELS = {
    "lax-friedrichs": "Lax-Friedrichs",
    "rusanov": "Rusanov",
    "hll": "HLL",
    "hllc": "HLLC",
    "roe": "Roe",
    "roe-nc": "Roe (no correction)",
    "godunov": "Godunov",
    "ausm+": "AUSM+",
    "ausm+-up": "AUSM+-up",
    "lax-wendroff": "Lax-Wendroff",
    "jst": "JST",
}

COMPOSABLE_FLUX = {"rusanov", "hll", "hllc", "roe", "roe-nc", "godunov", "ausm+", "ausm+-up"}

RECONSTRUCTIONS = {
    "none": "None",
    "muscl": "MUSCL (order 2)",
    "eno2": "ENO2 (order 2)",
    "weno3": "WENO3-JS (order 3)",
    "wenoz3": "WENO3-Z (order 3)",
    "weno5": "WENO5-JS (order 5)",
    "wenoz5": "WENO5-Z (order 5)",
}

RECON_ROWS = {
    "none": "Constant (order 1)",
    "muscl": "MUSCL (order 2)",
    "eno2": "ENO2 (order 2)",
    "weno3": "WENO3-JS (order 3)",
    "wenoz3": "WENO3-Z (order 3)",
    "weno5": "WENO5-JS (order 5)",
    "wenoz5": "WENO5-Z (order 5)",
}

COMPOSABLE_LIST = [f for f in ALL_FLUX if f in COMPOSABLE_FLUX]

CENTERED_SCHEMES = {
    "lax-friedrichs": "Lax-Friedrichs (order 1)",
    "lax-wendroff": "Lax-Wendroff (order 2)",
    "jst": "JST (order 2)",
}

DOCS_URL = "https://github.com/fhermet/euler-1d-solver/blob/main/docs/"

DOC_CHAPTERS = [
    ("1. Euler equations", "01_euler_equations.md"),
    ("2. Riemann problem, entropy", "02_riemann_problem.md"),
    ("3. Finite volumes", "03_finite_volume.md"),
    ("4. Numerical fluxes", "04_flux_schemes.md"),
    ("5. Reconstructions (MUSCL, ENO, WENO)", "05_reconstruction.md"),
    ("6. Time integration, CFL", "06_time_integration.md"),
    ("7. Fourier analysis", "07_fourier_analysis.md"),
    ("8. Test cases, convergence", "08_test_cases.md"),
]

GLOSSARY = [
    ("CFL", "Courant number Δt·max(|u|+c)/Δx. The solver caps it per scheme: "
     "1 for first-order schemes, 0.5 for MUSCL, ENO and WENO (§6.1)."),
    ("Effective CFL", "The CFL actually used, shown in the legends: the requested "
     "one, capped by the scheme's limit."),
    ("Auto (time integration)", "The scheme's default Runge-Kutta method: RK1 for "
     "first-order schemes, RK2 for MUSCL and ENO2, SSP-RK3 for WENO, RK4 for JST. "
     "Lax-Wendroff always uses RK1: its flux embeds the time step (§6.7)."),
    ("SSP", "Strong stability preserving: a Runge-Kutta method made of forward "
     "Euler steps, which keeps the non-oscillation property of the spatial scheme "
     "under the same CFL bound. RK1, Heun (RK2) and SSP-RK3 are SSP; RK4 and RK5 "
     "are not (§6.4)."),
    ("TVD", "Total variation diminishing: the scheme creates no new extremum. "
     "First-order upwind schemes and limited MUSCL (within its CFL bound) are TVD; "
     "ENO and WENO are only essentially non-oscillatory (§5.1 to §5.4)."),
    ("Order of convergence", "Slope of the error versus Δx in log-log scale. "
     "Expected only on smooth solutions; on discontinuous ones the L1 rate is "
     "about 1 or less whatever the scheme (§8.12)."),
    ("L1, L2, L∞", "Error norms: integral of |error|, root mean square, maximum. "
     "Near a discontinuity L∞ does not converge (§8.12)."),
    ("Local order", "Order between two successive meshes; it settles once the "
     "asymptotic regime is reached."),
    ("Exact (round-off)", "Errors below 1e-12: the scheme reproduces the solution, "
     "or the variable is constant. No order can be measured."),
    ("Reconstructed variables", "Variables on which MUSCL, ENO and WENO build the "
     "interface states: primitive (default, robust, order ≤ 2 on nonlinear flows), "
     "conservative or characteristic (keep the formal order) (§5.6)."),
    ("Positivity floor", "Option that clips negative density or pressure after "
     "each step. Not conservative, off by default, reported when used."),
    ("Diverged", "The run produced a negative density or pressure and stopped; "
     "the message says whether a smaller time step helps."),
    ("Entropy fix", "Correction of Roe's scheme at transonic rarefactions, "
     "without which it can keep an expansion shock (§2.2, case 'Sonic "
     "rarefaction')."),
    ("Expansion shock", "A discontinuous weak solution in place of a rarefaction: "
     "it satisfies Rankine-Hugoniot but decreases entropy, so it is not physical "
     "(§2.2)."),
    ("Reference solution", "Shu-Osher has no exact solution; its black curve is "
     "a WENO5-Z computation on 2000 cells."),
    ("σ (Fourier page)", "Courant number of the analysed wave, λΔt/Δx; smaller "
     "than the CFL except for the fastest wave (§7.5)."),
]


def documentation_sidebar() -> None:
    """Links to the theory chapters and a glossary of the terms used in the app."""
    with st.expander("Documentation (in French)"):
        for title, filename in DOC_CHAPTERS:
            st.markdown(f"- [{title}]({DOCS_URL}{filename})")
        st.caption("'§x.y' in the messages refers to these chapters.")
    with st.expander("Glossary"):
        for term, definition in GLOSSARY:
            st.markdown(f"**{term}**: {definition}")


RK_OPTIONS = ["Auto", "RK1", "RK2", "RK3", "RK4", "RK5"]

_RK_DEFAULTS = {
    "none": "RK1",
    "muscl": "RK2",
    "eno2": "RK2",
    "weno3": "RK3",
    "wenoz3": "RK3",
    "weno5": "RK3",
    "wenoz5": "RK3",
    "_centered": "Auto",
}

# ---------------------------------------------------------------------------
# Plotly thesis style
# ---------------------------------------------------------------------------

COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
MARKERS = [
    "circle", "square", "diamond", "triangle-up",
    "triangle-down", "cross", "x", "star",
]

_AXIS_COMMON = dict(
    showline=True,
    linewidth=1,
    linecolor="black",
    mirror=True,
    ticks="inside",
    tickwidth=1,
    ticklen=5,
    showgrid=True,
    gridcolor="#cccccc",
    gridwidth=0.5,
    tickfont=dict(family="STIX Two Text, serif", size=13),
    title=dict(font=dict(family="STIX Two Text, serif", size=15)),
)

THESIS_LAYOUT = dict(
    font=dict(family="STIX Two Text, serif", size=13),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=20, t=80, b=60),
)

_LEGEND_COMMON = dict(
    font=dict(family="STIX Two Text, serif", size=12),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#cccccc",
    borderwidth=1,
)


def thesis_axis(**overrides):
    """Return axis dict merged with thesis defaults."""
    ax = dict(_AXIS_COMMON)
    if "title_text" in overrides:
        text = overrides.pop("title_text")
        ax["title"] = dict(text=text, font=dict(family="STIX Two Text, serif", size=15))
    ax.update(overrides)
    return ax


# ---------------------------------------------------------------------------
# Scheme key builder
# ---------------------------------------------------------------------------

def build_scheme_key(flux: str, recon: str) -> str:
    """Build a registry key from flux solver + reconstruction names."""
    if recon == "none":
        return flux
    return f"{recon}-{flux}"


# ---------------------------------------------------------------------------
# Scheme selector widgets
# ---------------------------------------------------------------------------

def _build_grid_defaults() -> pd.DataFrame:
    """Build the default grid DataFrame for composable flux only."""
    rows = []
    for rk, rl in RECON_ROWS.items():
        row = {"Reconstruction": rl}
        for flux in COMPOSABLE_LIST:
            row[FLUX_LABELS[flux]] = (rk == "none" and flux == "hllc")
        rows.append(row)
    return pd.DataFrame(rows)


def scheme_selector_multi(key_prefix: str) -> list[tuple[str, str | None, str]]:
    """Multi-scheme selector as an interactive grid + limiter sub-grid + RK table.

    Returns list of (scheme_key, limiter_or_None, time_integrator).
    """
    grid_df = _build_grid_defaults()

    flux_col_config = {
        FLUX_LABELS[f]: st.column_config.CheckboxColumn(
            FLUX_LABELS[f], default=False,
        )
        for f in COMPOSABLE_LIST
    }

    edited = st.data_editor(
        grid_df,
        column_config={
            "Reconstruction": st.column_config.TextColumn(
                "Reconstruction", disabled=True, width="medium",
            ),
            **flux_col_config,
        },
        hide_index=True,
        use_container_width=True,
        key=f"{key_prefix}_grid",
    )

    # Extract selected combinations from main grid
    recon_keys = list(RECON_ROWS.keys())
    muscl_flux: list[str] = []
    used_recons: set[str] = set()

    non_muscl: list[tuple[str, str, None]] = []
    for i, rk in enumerate(recon_keys):
        for flux in COMPOSABLE_LIST:
            if bool(edited.iloc[i][FLUX_LABELS[flux]]):
                used_recons.add(rk)
                if rk == "muscl":
                    muscl_flux.append(flux)
                else:
                    non_muscl.append((build_scheme_key(flux, rk), rk, None))

    # --- Limiter sub-grid (only if MUSCL is used) ---
    muscl_schemes: list[tuple[str, str, str]] = []
    if muscl_flux:
        st.markdown(
            f"**MUSCL limiters** (applied to {', '.join(FLUX_LABELS[f] for f in muscl_flux)})"
        )
        lim_rows = []
        for lim_name in available_limiters():
            row = {"Limiter": lim_name}
            for flux in muscl_flux:
                row[FLUX_LABELS[flux]] = (lim_name == "van-leer")
            lim_rows.append(row)
        lim_df = pd.DataFrame(lim_rows)

        lim_col_config = {
            FLUX_LABELS[f]: st.column_config.CheckboxColumn(
                FLUX_LABELS[f], default=False,
            )
            for f in muscl_flux
        }

        edited_lim = st.data_editor(
            lim_df,
            column_config={
                "Limiter": st.column_config.TextColumn(
                    "Limiter", disabled=True, width="small",
                ),
                **lim_col_config,
            },
            hide_index=True,
            use_container_width=True,
            key=f"{key_prefix}_lim_grid",
        )

        for i, lim_name in enumerate(available_limiters()):
            for flux in muscl_flux:
                if bool(edited_lim.iloc[i][FLUX_LABELS[flux]]):
                    muscl_schemes.append(
                        (build_scheme_key(flux, "muscl"), "muscl", lim_name)
                    )

    # Centered schemes
    centered = st.multiselect(
        "Centered schemes",
        list(CENTERED_SCHEMES.keys()),
        format_func=lambda x: CENTERED_SCHEMES[x],
        key=f"{key_prefix}_centered",
    )
    if centered:
        used_recons.add("_centered")

    # --- Time integrator table ---
    st.markdown("**Time integration**")
    rk_rows = []
    rk_recon_order: list[str] = []
    for rk_key in list(RECON_ROWS.keys()) + ["_centered"]:
        if rk_key not in used_recons:
            continue
        rk_recon_order.append(rk_key)
        label = RECON_ROWS[rk_key] if rk_key != "_centered" else "Centered schemes"
        rk_rows.append({"Reconstruction": label, "RK": _RK_DEFAULTS[rk_key]})

    if rk_rows:
        rk_df = pd.DataFrame(rk_rows)
        edited_rk = st.data_editor(
            rk_df,
            column_config={
                "Reconstruction": st.column_config.TextColumn(
                    "Reconstruction", disabled=True, width="medium",
                ),
                "RK": st.column_config.SelectboxColumn(
                    "Time integration",
                    options=RK_OPTIONS,
                    width="small",
                ),
            },
            hide_index=True,
            use_container_width=True,
            key=f"{key_prefix}_rk_grid",
        )
        rk_map = {rk_recon_order[i]: edited_rk.iloc[i]["RK"] for i in range(len(rk_recon_order))}
    else:
        rk_map = {}

    # Build final list
    result: list[tuple[str, str | None, str]] = []
    for key, recon_key, _ in non_muscl:
        ti = rk_map.get(recon_key, "Auto")
        result.append((key, None, ti))
    for key, recon_key, lim in muscl_schemes:
        ti = rk_map.get("muscl", "Auto")
        result.append((key, lim, ti))
    for key in centered:
        ti = rk_map.get("_centered", "Auto")
        result.append((key, None, ti))

    if result:
        names = []
        for key, lim, ti in result:
            s = get_scheme(key, **({"limiter": lim} if lim else {}))
            rk = select_time_integrator(s, None if ti == "Auto" else ti)
            forced = s.time_integral_included and ti not in ("Auto", "RK1")
            note = ", time integration built into the flux" if forced else ""
            names.append(f"{s.name} [{rk}{note}]")
        st.caption("**" + str(len(result)) + " scheme(s)**: " + " · ".join(names))

    return result


def scheme_selector_single(key_prefix: str) -> tuple[str, str | None, str]:
    """Single-scheme selector: flux + reconstruction + limiter + RK.

    Returns (scheme_key, limiter_or_None, time_integrator).
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        flux = st.selectbox(
            "Numerical flux",
            ALL_FLUX,
            index=3,  # hllc
            format_func=lambda x: FLUX_LABELS[x],
            key=f"{key_prefix}_flux",
        )
    with col2:
        if flux in COMPOSABLE_FLUX:
            recon = st.selectbox(
                "Reconstruction",
                list(RECONSTRUCTIONS.keys()),
                format_func=lambda x: RECONSTRUCTIONS[x],
                key=f"{key_prefix}_recon",
            )
        else:
            st.selectbox(
                "Reconstruction",
                ["None"],
                key=f"{key_prefix}_recon",
                disabled=True,
            )
            recon = "none"
    limiter = None
    if recon == "muscl":
        with col3:
            limiter = st.selectbox(
                "Limiter",
                available_limiters(),
                key=f"{key_prefix}_lim",
            )
    with col4:
        ti = st.selectbox(
            "Time integration",
            RK_OPTIONS,
            index=0,
            key=f"{key_prefix}_rk",
        )
    return build_scheme_key(flux, recon), limiter, ti
