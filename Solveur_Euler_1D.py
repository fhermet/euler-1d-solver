#!/usr/bin/env python3
"""Interface graphique Streamlit pour le solveur Euler 1D."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from euler1d.config import (
    GasProperties,
    MeshConfig,
    RiemannProblem,
    SimulationConfig,
    SmoothProblem,
    TimeConfig,
)
from euler1d.physics import conservative_to_primitive
from euler1d.results import (
    compute_exact_solution,
    convergence_study,
    estimate_order,
)
from euler1d.riemann import sample_riemann
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from ui_common import (
    COLORS,
    MARKERS,
    THESIS_LAYOUT,
    _LEGEND_COMMON,
    scheme_selector_multi,
    scheme_selector_single,
    thesis_axis,
)
from euler1d.test_cases import (
    acoustic_wave,
    double_rarefaction,
    entropy_wave,
    lax_test,
    near_vacuum,
    shu_osher,
    sod_shock_tube,
    stationary_contact,
    two_shocks,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_CASES = {
    "Sod shock tube": sod_shock_tube,
    "Lax": lax_test,
    "Double rarefaction": double_rarefaction,
    "Contact stationnaire": stationary_contact,
    "Quasi-vide": near_vacuum,
    "Two shocks (Toro 4)": two_shocks,
    "Shu-Osher": shu_osher,
    "Entropy wave": entropy_wave,
    "Acoustic wave": acoustic_wave,
}

SMOOTH_CASES = {"Entropy wave", "Acoustic wave"}






TEST_CASE_INFO = {
    "Sod shock tube": {
        "description": (
            "Probleme de Riemann classique avec une discontinuite initiale de pression "
            "et de densite a $x = 0.5$. Etat gauche : $(\\rho, u, p) = (1, 0, 1)$, "
            "etat droit : $(\\rho, u, p) = (0.125, 0, 0.1)$. "
            "Ce cas est le test de reference pour valider un solveur de Riemann."
        ),
        "structure": (
            "**Structure de la solution :** onde de detente a gauche, "
            "discontinuite de contact au centre, onde de choc a droite."
        ),
    },
    "Lax": {
        "description": (
            "Probleme de Riemann plus severe avec une vitesse non nulle a gauche. "
            "Etat gauche : $(\\rho, u, p) = (0.445, 0.698, 3.528)$, "
            "etat droit : $(\\rho, u, p) = (0.5, 0, 0.571)$. "
            "Teste la robustesse des schemas sur des gradients plus raides."
        ),
        "structure": (
            "**Structure de la solution :** onde de detente a gauche, "
            "discontinuite de contact, onde de choc a droite. "
            "Les ondes sont plus intenses que dans le cas Sod."
        ),
    },
    "Double rarefaction": {
        "description": (
            "Vitesses initiales symetriques et opposees : le fluide s'ecarte du centre. "
            "Etat gauche : $(\\rho, u, p) = (1, -2, 0.4)$, "
            "etat droit : $(\\rho, u, p) = (1, 2, 0.4)$. "
            "Teste la capacite du schema a gerer les zones de quasi-vide (basse densite/pression au centre)."
        ),
        "structure": (
            "**Structure de la solution :** deux ondes de detente s'eloignant "
            "l'une de l'autre, avec une zone de basse densite/pression au centre."
        ),
    },
    "Contact stationnaire": {
        "description": (
            "Discontinuite de contact immobile : seule la densite varie. "
            "Etat gauche : $(\\rho, u, p) = (1, 0, 1)$, "
            "etat droit : $(\\rho, u, p) = (0.125, 0, 1)$. "
            "La solution exacte est stationnaire : tout etalement du profil de densite "
            "est purement du a la diffusion numerique du schema."
        ),
        "structure": (
            "**Structure de la solution :** une seule discontinuite de contact "
            "immobile a $x = 0.5$. Pas de choc, pas de detente. "
            "Les schemas resolvant l'onde de contact (HLLC, Roe, Godunov) "
            "n'introduisent aucune diffusion. HLL et Rusanov etalent le contact."
        ),
    },
    "Quasi-vide": {
        "description": (
            "Double detente extreme proche de la limite de formation du vide. "
            "Etat gauche : $(\\rho, u, p) = (1, -3.5, 0.4)$, "
            "etat droit : $(\\rho, u, p) = (1, 3.5, 0.4)$. "
            "Les vitesses sont a 93% du seuil de vide : la pression au centre "
            "atteint $p^* \\sim 10^{-13}$, testant la preservation de positivite des schemas."
        ),
        "structure": (
            "**Structure de la solution :** deux ondes de detente symetriques "
            "avec une zone de densite/pression quasi nulle au centre. "
            "Version extreme de la double detente ($u = \\pm 3.5$ au lieu de $\\pm 2$)."
        ),
    },
    "Two shocks (Toro 4)": {
        "description": (
            "Collision de deux chocs forts (Toro test 4). "
            "Etat gauche : $(\\rho, u, p) = (5.99924, 19.5975, 460.894)$, "
            "etat droit : $(\\rho, u, p) = (5.99242, -6.19633, 46.0950)$. "
            "Les vitesses convergentes creent une zone de tres haute pression ($p^* \\approx 1700$) au centre."
        ),
        "structure": (
            "**Structure de la solution :** choc a gauche, "
            "discontinuite de contact au centre, choc a droite. "
            "C'est le symetrique de la double detente : compression au lieu d'expansion."
        ),
    },
    "Shu-Osher": {
        "description": (
            "Interaction choc/onde entropique. Un choc Mach 3 se propage dans un milieu "
            "a densite sinusoidale : $\\rho_R = 1 + 0.2\\sin(5x)$. "
            "Derriere le choc, des oscillations physiques de petite echelle apparaissent. "
            "La solution de reference est calculee a haute resolution (WENO5-Z, 2000 cellules)."
        ),
        "structure": (
            "**Structure de la solution :** choc principal a droite, "
            "suivi d'une zone complexe d'oscillations haute frequence. "
            "MUSCL lisse ces oscillations, WENO5 les capture fidelement. "
            "C'est le meilleur cas pour montrer l'apport des schemas d'ordre eleve."
        ),
    },
    "Entropy wave": {
        "description": (
            "Perturbation sinusoidale de densite advectee a vitesse constante $u_0 = 1$. "
            "La pression est uniforme et la vitesse constante : seule la densite varie. "
            "Solution lisse, ideale pour verifier l'ordre de convergence des schemas."
        ),
        "structure": (
            "**Structure de la solution :** la perturbation de densite est simplement "
            "translatee sans deformation. Les champs $u$ et $p$ restent constants."
        ),
    },
    "Acoustic wave": {
        "description": (
            "Petite perturbation isentropique ($\\epsilon = 10^{-4}$) se propageant "
            "a la vitesse du son. Solution lisse linearisee, valide pour les petites amplitudes. "
            "Teste la convergence sur les champs genuinement non-lineaires ($\\rho$, $u$, $p$ varient tous)."
        ),
        "structure": (
            "**Structure de la solution :** onde acoustique se propageant vers la droite "
            "a la vitesse $u_0 + c_0$. Tous les champs presentent une perturbation sinusoidale."
        ),
    },
}

VAR_LABELS = {
    "rho": "ρ (densité)",
    "u": "u (vitesse)",
    "p": "p (pression)",
}
VAR_LATEX = {"rho": "ρ", "u": "u", "p": "p"}

NORMS = ("L1", "L2", "Linf")
NORM_LATEX = {"L1": "L₁", "L2": "L₂", "Linf": "L∞"}

# Style constants imported from ui_common:
# COLORS, MARKERS, THESIS_LAYOUT, _LEGEND_COMMON, thesis_axis


def _make_initial_conditions_figure(
    test_case_name: str, n_cells: int, gamma: float,
) -> go.Figure:
    """Create 3 subplots showing initial conditions (rho, u, p) at t=0."""
    config = TEST_CASES[test_case_name](n_cells=n_cells)
    mesh = config.mesh
    dx = mesh.dx
    x = np.linspace(mesh.x_min + 0.5 * dx, mesh.x_max - 0.5 * dx, mesh.n_cells)
    prob = config.problem

    if isinstance(prob, SmoothProblem):
        rho, u, p = prob.exact_fn(x, 0.0, config.gas.gamma)
    elif isinstance(prob, RiemannProblem):
        rho = np.where(x < prob.x_discontinuity, prob.left.rho, prob.right.rho)
        u = np.where(x < prob.x_discontinuity, prob.left.u, prob.right.u)
        p = np.where(x < prob.x_discontinuity, prob.left.p, prob.right.p)
    else:
        raise ValueError(f"Unknown problem type: {type(prob)}")

    variables = ("rho", "u", "p")
    arrays = (rho, u, p)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[VAR_LABELS[v] for v in variables],
        horizontal_spacing=0.08,
    )

    for j, (var, arr) in enumerate(zip(variables, arrays), 1):
        fig.add_trace(go.Scatter(
            x=x, y=arr, mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        ), row=1, col=j)
        fig.update_xaxes(thesis_axis(title_text="x"), row=1, col=j)
        fig.update_yaxes(thesis_axis(title_text=VAR_LATEX[var]), row=1, col=j)

    fig.update_layout(
        **THESIS_LAYOUT,
        title=dict(
            text=f"Conditions initiales — {test_case_name}",
            font=dict(family="STIX Two Text, serif", size=17), x=0.0, xanchor="left",
        ),
        height=350, width=1200,
    )
    fig.update_annotations(font=dict(family="STIX Two Text, serif", size=15), yshift=15)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(test_case_name: str, n_cells: int, cfl: float, gamma: float) -> SimulationConfig:
    """Reconstruct a SimulationConfig from UI parameters."""
    base = TEST_CASES[test_case_name](n_cells=n_cells)
    if test_case_name in SMOOTH_CASES:
        return base._replace() if hasattr(base, "_replace") else SimulationConfig(
            gas=base.gas,
            mesh=base.mesh,
            time=TimeConfig(t_final=base.time.t_final, cfl=cfl),
            problem=base.problem,
            bc_type=base.bc_type,
        )
    return SimulationConfig(
        gas=GasProperties(gamma=gamma),
        mesh=MeshConfig(x_min=base.mesh.x_min, x_max=base.mesh.x_max, n_cells=n_cells),
        time=TimeConfig(t_final=base.time.t_final, cfl=cfl),
        problem=base.problem,
        bc_type=base.bc_type,
    )


def _compute_exact_at_time(
    config: SimulationConfig, t: float
) -> pd.DataFrame:
    """Compute exact solution at arbitrary time t."""
    mesh = config.mesh
    gas = config.gas
    prob = config.problem
    dx = mesh.dx
    x = np.linspace(mesh.x_min + 0.5 * dx, mesh.x_max - 0.5 * dx, mesh.n_cells)

    if isinstance(prob, SmoothProblem):
        rho, u, p = prob.exact_fn(x, t, gas.gamma)
    elif isinstance(prob, RiemannProblem):
        rho, u, p = sample_riemann(
            x, t,
            prob.left.rho, prob.left.u, prob.left.p,
            prob.right.rho, prob.right.u, prob.right.p,
            gas.gamma,
            x0=prob.x_discontinuity,
        )
    else:
        raise ValueError(f"Unknown problem type: {type(prob)}")

    return pd.DataFrame({"x": x, "rho": rho, "u": u, "p": p})


# ---------------------------------------------------------------------------
# Cached simulation functions
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_simulate(
    test_case_name: str,
    n_cells: int,
    cfl: float,
    gamma: float,
    scheme_name: str,
    limiter: str | None,
    n_snapshots: int,
    time_integrator: str = "Auto",
) -> tuple[list, list, float, str]:
    """Run a simulation and return (snapshots_as_lists, times, wall_time, display_name)."""
    config = _build_config(test_case_name, n_cells, cfl, gamma)
    kwargs = {"limiter": limiter} if limiter and scheme_name.startswith("muscl") else {}
    scheme = get_scheme(scheme_name, **kwargs)
    rk = None if time_integrator == "Auto" else time_integrator
    result = run_simulation(config, scheme, n_snapshots=n_snapshots, time_integrator=rk)
    snapshots = [s.tolist() for s in result.snapshots]
    return snapshots, result.times, result.wall_time, scheme.name


@st.cache_data(show_spinner=False)
def cached_exact(
    test_case_name: str, n_cells: int, cfl: float, gamma: float
) -> pd.DataFrame:
    """Compute exact solution at t_final."""
    config = _build_config(test_case_name, n_cells, cfl, gamma)
    return compute_exact_solution(config)


@st.cache_data(show_spinner=False)
def cached_convergence(
    test_case_name: str,
    cfl: float,
    gamma: float,
    scheme_name: str,
    limiter: str | None,
    n_cells_tuple: tuple[int, ...],
    time_integrator: str = "Auto",
) -> pd.DataFrame:
    """Run convergence study for one scheme."""
    make_config_fn = TEST_CASES[test_case_name]
    # Wrap to inject cfl/gamma
    if test_case_name in SMOOTH_CASES:
        def make_config(n):
            base = make_config_fn(n_cells=n)
            return SimulationConfig(
                gas=base.gas, mesh=base.mesh,
                time=TimeConfig(t_final=base.time.t_final, cfl=cfl),
                problem=base.problem, bc_type=base.bc_type,
            )
    else:
        def make_config(n):
            base = make_config_fn(n_cells=n)
            return SimulationConfig(
                gas=GasProperties(gamma=gamma),
                mesh=MeshConfig(x_min=base.mesh.x_min, x_max=base.mesh.x_max, n_cells=n),
                time=TimeConfig(t_final=base.time.t_final, cfl=cfl),
                problem=base.problem, bc_type=base.bc_type,
            )

    kwargs = {"limiter": limiter} if limiter and scheme_name.startswith("muscl") else {}
    scheme = get_scheme(scheme_name, **kwargs)
    rk = None if time_integrator == "Auto" else time_integrator
    return convergence_study(make_config, scheme, list(n_cells_tuple), time_integrator=rk)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _make_profile_figure(
    dfs: list[tuple[str, pd.DataFrame]],
    exact: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """Create 3 subplots (rho, u, p) comparing numerical vs exact."""
    variables = ("rho", "u", "p")
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[VAR_LABELS[v] for v in variables],
        horizontal_spacing=0.08,
    )

    for j, var in enumerate(variables, 1):
        showlegend = j == 1
        fig.add_trace(go.Scatter(
            x=exact["x"], y=exact[var], mode="lines",
            line=dict(color="black", width=2),
            name="Exacte", legendgroup="Exacte", showlegend=showlegend,
        ), row=1, col=j)
        for i, (name, df) in enumerate(dfs):
            fig.add_trace(go.Scatter(
                x=df["x"], y=df[var], mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                marker=dict(
                    symbol=MARKERS[i % len(MARKERS)],
                    size=5, maxdisplayed=30,
                ),
                name=name, legendgroup=name, showlegend=showlegend,
            ), row=1, col=j)

        fig.update_xaxes(thesis_axis(title_text="x"), row=1, col=j)
        fig.update_yaxes(thesis_axis(title_text=VAR_LATEX[var]), row=1, col=j)

    fig.update_layout(
        **THESIS_LAYOUT,
        title=dict(text=title, font=dict(family="STIX Two Text, serif", size=17), x=0.0, xanchor="left"),
        height=450, width=1200,
        legend=dict(
            **_LEGEND_COMMON,
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
        ),
    )
    fig.update_annotations(font=dict(family="STIX Two Text, serif", size=15), yshift=15)
    return fig


def _make_convergence_figure(
    conv_dfs: list[tuple[str, pd.DataFrame]],
    variable: str,
    title: str = "",
) -> go.Figure:
    """Create 3 log-log subplots for L1, L2, Linf convergence."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Erreur {NORM_LATEX[n]}" for n in NORMS],
        horizontal_spacing=0.08,
    )

    ref_styles = [
        (0.5, "dot", "O(Δx⁰·⁵)"),
        (1.0, "dash", "O(Δx)"),
        (2.0, "dashdot", "O(Δx²)"),
    ]

    for col_idx, norm in enumerate(NORMS, 1):
        showlegend = col_idx == 1

        for i, (name, df) in enumerate(conv_dfs):
            var_data = df[df["variable"] == variable]
            dx_vals = var_data["dx"].values
            err = var_data[norm].values
            if len(dx_vals) < 2:
                continue
            order = estimate_order(dx_vals, err)
            fig.add_trace(go.Scatter(
                x=dx_vals, y=err, mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                marker=dict(size=6),
                name=f"{name} (p={order:.2f})",
                legendgroup=f"{name}_{norm}",
                showlegend=showlegend,
            ), row=1, col=col_idx)

        # Reference slopes
        if conv_dfs:
            all_dx, all_err = [], []
            for _, df in conv_dfs:
                vd = df[df["variable"] == variable]
                if len(vd) > 0:
                    all_dx.extend(vd["dx"].values)
                    all_err.extend(vd[norm].values)
            if all_dx:
                dx_ref = np.array([min(all_dx), max(all_dx)])
                e_anchor = np.median(all_err) * 0.5
                for slope, dash, label in ref_styles:
                    fig.add_trace(go.Scatter(
                        x=dx_ref,
                        y=e_anchor * (dx_ref / dx_ref[0]) ** slope,
                        mode="lines",
                        line=dict(color="grey", width=1, dash=dash),
                        name=label, legendgroup=f"ref_{slope}",
                        showlegend=showlegend,
                    ), row=1, col=col_idx)

        fig.update_xaxes(
            thesis_axis(title_text="Δx", type="log", exponentformat="power"),
            row=1, col=col_idx,
        )
        fig.update_yaxes(
            thesis_axis(title_text="Erreur", type="log", exponentformat="power"),
            row=1, col=col_idx,
        )

    fig.update_layout(
        **THESIS_LAYOUT,
        title=dict(text=title, font=dict(family="STIX Two Text, serif", size=17), x=0.0, xanchor="left"),
        height=500, width=1200,
        legend=dict(
            **_LEGEND_COMMON,
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
        ),
    )
    fig.update_annotations(font=dict(family="STIX Two Text, serif", size=15), yshift=15)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Solveur Euler 1D", layout="wide")
st.title("Solveur Euler 1D -- Interface interactive")

with st.sidebar:
    st.header("Configuration")
    test_case_name = st.selectbox("Cas test", list(TEST_CASES.keys()))
    is_smooth = test_case_name in SMOOTH_CASES

    with st.expander("Parametres avances"):
        n_cells = st.slider("Nombre de cellules", 50, 2000, 200, step=10)
        cfl = st.slider("CFL", 0.1, 1.0, 0.9, step=0.05)
        if is_smooth:
            gamma = 1.4
            st.info("Gamma fixe a 1.4 pour les cas lisses")
        else:
            gamma = st.slider("Gamma", 1.1, 2.0, 1.4, step=0.05)


# ---------------------------------------------------------------------------
# Test case description
# ---------------------------------------------------------------------------

with st.expander(f"Description du cas : {test_case_name}", expanded=False):
    info = TEST_CASE_INFO[test_case_name]
    st.markdown(info["description"])
    st.markdown(info["structure"])
    fig_ic = _make_initial_conditions_figure(test_case_name, n_cells, gamma)
    st.plotly_chart(fig_ic, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab4 = st.tabs([
    "Ordre de convergence",
    "Evolution temporelle",
])


# ---- Tab 1: Scheme comparison ----
with tab1:
    st.subheader("Comparaison de schemas")
    selected = scheme_selector_multi("tab1")

    if selected:
        with st.spinner("Simulation en cours..."):
            dfs_list = []
            exact = cached_exact(test_case_name, n_cells, cfl, gamma)

            for sname, limiter, ti in selected:
                snapshots, times, wall_time, display_name = cached_simulate(
                    test_case_name, n_cells, cfl, gamma, sname, limiter, 1, ti,
                )
                if not snapshots:
                    st.warning(f"**{display_name}** : simulation divergente (instabilite numerique).")
                    continue
                U = np.array(snapshots[-1])
                rho, u, p = conservative_to_primitive(U, gamma)
                x = exact["x"].values
                df_num = pd.DataFrame({"x": x, "rho": rho, "u": u, "p": p})
                dfs_list.append((display_name, df_num))

            fig = _make_profile_figure(
                dfs_list, exact,
                title=f"{test_case_name} — N={n_cells}, CFL={cfl}",
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Convergence section ---
        st.markdown("#### Convergence")
        col_a, col_b = st.columns(2)
        with col_a:
            resolutions_str = st.text_input(
                "Resolutions (separees par des espaces)",
                "50 100 200 400 800",
                key="tab1_res",
            )
        with col_b:
            conv_var = st.selectbox("Variable", ["rho", "u", "p"], key="tab1_var")

        try:
            n_cells_list = tuple(sorted(int(x) for x in resolutions_str.split()))
        except ValueError:
            st.error("Resolutions invalides. Entrez des entiers separes par des espaces.")
            n_cells_list = ()

        if n_cells_list:
            with st.spinner("Convergence en cours..."):
                conv_dfs = []
                order_rows = []
                for sname, limiter, ti in selected:
                    df = cached_convergence(test_case_name, cfl, gamma, sname, limiter, n_cells_list, ti)
                    if df.empty:
                        continue
                    display_name = df["scheme"].iloc[0] if len(df) > 0 else sname
                    conv_dfs.append((display_name, df))

                    var_data = df[df["variable"] == conv_var]
                    if len(var_data) >= 2:
                        dx_vals = var_data["dx"].values
                        row = {"Schema": display_name}
                        for norm in NORMS:
                            err = var_data[norm].values
                            row[f"Ordre ({norm})"] = f"{estimate_order(dx_vals, err):.2f}"
                        order_rows.append(row)

                fig = _make_convergence_figure(
                    conv_dfs, conv_var,
                    title=f"Convergence — {test_case_name} ({VAR_LATEX[conv_var]})",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Ordres de convergence estimes")
                st.dataframe(pd.DataFrame(order_rows), width="stretch", hide_index=True)
    else:
        st.info("Selectionnez au moins un flux et une reconstruction.")



# ---- Tab 4: Temporal animation ----
with tab4:
    st.subheader("Animation temporelle")

    anim_scheme, anim_limiter, anim_ti = scheme_selector_single("tab4")
    n_snapshots = st.slider("Nombre de snapshots", 10, 200, 50, key="tab4_snaps")

    with st.spinner("Simulation..."):
        snapshots, snap_times, wt, dn = cached_simulate(
            test_case_name, n_cells, cfl, gamma, anim_scheme,
            anim_limiter, n_snapshots, anim_ti,
        )

    if not snapshots:
        st.error(f"**{dn}** : simulation divergente (instabilite numerique).")
        st.stop()

    config = _build_config(test_case_name, n_cells, cfl, gamma)
    t_final = config.time.t_final

    # Time slider
    if len(snap_times) > 1:
        time_options = [f"{t:.6f}" for t in snap_times]
        selected_time_str = st.select_slider(
            "Temps",
            options=time_options,
            value=time_options[-1],
            key="tab4_time",
        )
        time_idx = time_options.index(selected_time_str)
        selected_time = snap_times[time_idx]
    else:
        time_idx = 0
        selected_time = snap_times[0] if snap_times else t_final

    st.write(f"**Schema :** {dn} | **t =** {selected_time:.6f} / {t_final:.6f} | **Temps de calcul :** {wt:.4f} s")

    # Plot snapshot vs exact at that time
    U = np.array(snapshots[time_idx])
    rho_num, u_num, p_num = conservative_to_primitive(U, gamma)
    mesh = config.mesh
    dx = mesh.dx
    x = np.linspace(mesh.x_min + 0.5 * dx, mesh.x_max - 0.5 * dx, mesh.n_cells)

    exact_t = _compute_exact_at_time(config, selected_time)

    variables = ("rho", "u", "p")
    num_arrays = (rho_num, u_num, p_num)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[VAR_LABELS[v] for v in variables],
        horizontal_spacing=0.08,
    )

    for j, (var, num_data) in enumerate(zip(variables, num_arrays), 1):
        showlegend = j == 1
        fig.add_trace(go.Scatter(
            x=exact_t["x"], y=exact_t[var], mode="lines",
            line=dict(color="black", width=2),
            name="Exacte", legendgroup="Exacte", showlegend=showlegend,
        ), row=1, col=j)
        fig.add_trace(go.Scatter(
            x=x, y=num_data, mode="lines+markers",
            line=dict(color=COLORS[0], width=1.5),
            marker=dict(symbol=MARKERS[0], size=5, maxdisplayed=30),
            name=dn, legendgroup=dn, showlegend=showlegend,
        ), row=1, col=j)

        fig.update_xaxes(thesis_axis(title_text="x"), row=1, col=j)
        fig.update_yaxes(thesis_axis(title_text=VAR_LATEX[var]), row=1, col=j)

    fig.update_layout(
        **THESIS_LAYOUT,
        title=dict(
            text=f"{test_case_name} — {dn} à t={selected_time:.4f}",
            font=dict(family="STIX Two Text, serif", size=17), x=0.0, xanchor="left",
        ),
        height=450, width=1200,
        legend=dict(
            **_LEGEND_COMMON,
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
        ),
    )
    fig.update_annotations(font=dict(family="STIX Two Text, serif", size=15), yshift=15)
    st.plotly_chart(fig, use_container_width=True)

