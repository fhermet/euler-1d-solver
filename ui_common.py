"""Shared UI constants, styles, and scheme selector widgets for Streamlit pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from euler1d.schemes import get_scheme
from euler1d.schemes.reconstruction import available_limiters

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

RK_OPTIONS = ["Auto", "RK1", "RK2", "RK3", "RK4", "RK5"]

_RK_DEFAULTS = {
    "none": "RK1",
    "muscl": "RK2",
    "eno2": "RK2",
    "weno3": "RK3",
    "wenoz3": "RK3",
    "weno5": "RK5",
    "wenoz5": "RK5",
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
            rk_label = f" [{ti}]" if ti != "Auto" else ""
            names.append(f"{s.name}{rk_label}")
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
