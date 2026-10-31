"""Home page: the finite-volume method, step by step, on the Sod shock tube.

Follows one time step of the solver cell by cell. Every number shown comes
from euler1d.fv_walkthrough, which calls the solver's own functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from euler1d.fv_walkthrough import (
    advance, default_interface, exact_waves, flat_cells, flux_breakdown, global_balance,
    interface_states, model_waves, rk_stages, sod_setup, stage_update, time_step,
)
from euler1d.physics import conservative_to_primitive
from euler1d.results import exact_cell_averages
from euler1d.schemes import get_scheme
from euler1d.schemes.reconstruction import available_limiters
from euler1d.solver import initial_state, select_time_integrator
from ui_common import DOCS_URL, FLUX_LABELS, RK_OPTIONS, documentation_sidebar
from ui_walkthrough import (
    COMPONENT_NAMES, VAR_NAMES, fig_averages, fig_fluxes, fig_reconstruction, fig_riemann,
    fig_stages, fig_update,
)

st.set_page_config(page_title="Comprendre les volumes finis", layout="wide")

with st.sidebar:
    documentation_sidebar()

st.title("Comprendre la méthode des volumes finis")
st.markdown(
    "Cette page suit **un pas de temps du solveur, maille par maille**, sur le tube de Sod. "
    "Chaque étape montre ce que calcule le vrai code, pour le flux et la reconstruction "
    "choisis. Les pages **Solveur** et **Analyse de Fourier** servent ensuite à "
    "expérimenter."
)

STEPS = ["Moyennes", "Reconstruction", "Problème de Riemann", "Flux", "Mise à jour",
         "Runge-Kutta et CFL"]
FLUXES = ["rusanov", "hll", "hllc", "roe", "roe-nc", "godunov", "ausm+", "ausm+-up"]
FLUX_NAMES = {**FLUX_LABELS, "roe-nc": "Roe sans correction d'entropie"}
RECONSTRUCTIONS = {
    "constant": "Constante (ordre 1)",
    "muscl": "MUSCL (ordre 2)",
    "eno2": "ENO2 (ordre 2)",
    "weno3": "WENO3 (ordre 3)",
    "weno5": "WENO5 (ordre 5)",
}


def doc_link(filename: str, label: str) -> str:
    return f"[{label}]({DOCS_URL}{filename})"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ss = st.session_state


def kept(key: str, default):
    """Current value of the widget *key*, kept in a key of its own.

    Streamlit deletes a widget's key when the widget is not drawn in a run
    (another step, another page): each widget reads its value from
    'keep_<key>' and writes it back there when the student changes it.
    """
    return ss.setdefault("keep_" + key, default)


def save(key: str) -> None:
    ss["keep_" + key] = ss[key]


c1, c2, c3, c4 = st.columns(4)
with c1:
    flux = st.selectbox("Flux numérique", FLUXES, index=FLUXES.index(kept("fv_flux", "hllc")),
                        format_func=FLUX_NAMES.get, key="fv_flux", on_change=save,
                        args=("fv_flux",))
    rk_choice = st.selectbox("Intégrateur en temps", RK_OPTIONS,
                             index=RK_OPTIONS.index(kept("fv_rk", "Auto")), key="fv_rk",
                             on_change=save, args=("fv_rk",))
with c2:
    recon = st.selectbox("Reconstruction", list(RECONSTRUCTIONS),
                         index=list(RECONSTRUCTIONS).index(kept("fv_recon", "muscl")),
                         format_func=RECONSTRUCTIONS.get, key="fv_recon", on_change=save,
                         args=("fv_recon",))
    limiter = None
    if recon == "muscl":
        limiter = st.selectbox("Limiteur", available_limiters(),
                               index=available_limiters().index(kept("fv_limiter", "van-leer")),
                               key="fv_limiter", on_change=save, args=("fv_limiter",))
with c3:
    var = VAR_NAMES.index(st.radio("Variable affichée", VAR_NAMES, horizontal=True,
                                   index=VAR_NAMES.index(kept("fv_var", "ρ")), key="fv_var",
                                   on_change=save, args=("fv_var",)))
    n_cells = st.slider("Nombre de mailles", 8, 32, kept("fv_cells", 16), key="fv_cells",
                        on_change=save, args=("fv_cells",))
with c4:
    cfl = st.slider("CFL demandé", 0.1, 1.0, kept("fv_cfl", 0.9), step=0.05, key="fv_cfl",
                    on_change=save, args=("fv_cfl",))
    show_exact = st.checkbox("Superposer la solution exacte", value=kept("fv_exact", True),
                             key="fv_exact", on_change=save, args=("fv_exact",))

scheme_key = flux if recon == "constant" else f"{recon}-{flux}"
scheme = get_scheme(scheme_key, **({"limiter": limiter} if limiter else {}))
rk = None if rk_choice == "Auto" else rk_choice
config = sod_setup(n_cells, cfl)
gamma, dx = config.gas.gamma, config.mesh.dx
x = (np.arange(n_cells) + 0.5) * dx

# Changing a setting restarts at t = 0 but keeps the step shown.
settings = (scheme_key, limiter, rk, n_cells, cfl)
ss.setdefault("fv_step", 1)
if ss.get("fv_settings") != settings:
    ss.fv_settings = settings
    ss.fv_U = initial_state(config)[1]
    ss.fv_t, ss.fv_n, ss.fv_diverged, ss.fv_half_ok = 0.0, 0, False, None


# ---------------------------------------------------------------------------
# Time advance
# ---------------------------------------------------------------------------

def _advance(n_steps: int) -> None:
    a = advance(ss.fv_U, scheme, config, rk, n_steps)
    ss.fv_U, ss.fv_t, ss.fv_n = a.U, ss.fv_t + a.elapsed, ss.fv_n + a.steps_done
    ss.fv_diverged = a.diverged
    if a.diverged:
        # Same diagnosis as the solver page: does half the CFL help?
        half = sod_setup(n_cells, time_step(a.U, scheme, config).cfl / 2)
        ss.fv_half_ok = not advance(a.U, scheme, half, rk, 1).diverged


def _reset() -> None:
    ss.fv_U = initial_state(config)[1]
    ss.fv_t, ss.fv_n, ss.fv_diverged, ss.fv_half_ok = 0.0, 0, False, None


t1, t2, t3, t4 = st.columns([1, 1, 1, 3])
t1.button("Pas de temps suivant", key="fv_step_btn", on_click=_advance, args=(1,),
          disabled=ss.fv_diverged, width="stretch")
t2.button("+10 pas", key="fv_ten_btn", on_click=_advance, args=(10,),
          disabled=ss.fv_diverged, width="stretch")
t3.button("Réinitialiser", key="fv_reset_btn", on_click=_reset, width="stretch")
t4.markdown(f"**t = {ss.fv_t:.4f}**, pas n° {ss.fv_n} — {scheme.name}")

if ss.fv_diverged:
    cause = (
        "À CFL moitié, le même pas réussit : c'est une limite de pas de temps du schéma "
        "sur ce cas (exemple : AUSM+ sur Sod, §4.9)."
        if ss.fv_half_ok else
        "Il échoue encore à CFL moitié : le pas de temps n'est pas en cause, c'est le "
        "schéma spatial qui produit un état non physique ici (§8.5)."
    )
    st.error(
        f"**{scheme.name}** : divergence au pas n° {ss.fv_n + 1} (densité ou pression "
        f"négative). {cause} Les étapes montrent le dernier état physique ; "
        f"« Réinitialiser » pour repartir."
    )


# ---------------------------------------------------------------------------
# Step navigation
# ---------------------------------------------------------------------------

def _goto(step: int) -> None:
    ss.fv_step = min(max(step, 1), len(STEPS))


cols = st.columns(len(STEPS) + 2)
cols[0].button("◀ Précédent", key="fv_prev", on_click=_goto, args=(ss.fv_step - 1,),
               disabled=ss.fv_step == 1, width="stretch")
for s, name in enumerate(STEPS, 1):
    cols[s].button(f"{s}. {name}", key=f"fv_goto_{s}", on_click=_goto, args=(s,),
                   type="primary" if s == ss.fv_step else "secondary",
                   width="stretch")
cols[-1].button("Suivant ▶", key="fv_next", on_click=_goto, args=(ss.fv_step + 1,),
                disabled=ss.fv_step == len(STEPS), width="stretch")


# ---------------------------------------------------------------------------
# Quantities of the current step (state U^n)
# ---------------------------------------------------------------------------

U = ss.fv_U
ts = time_step(U, scheme, config)
UL, UR = interface_states(U, scheme, gamma, dx)
exact = None
if show_exact:
    exact = np.array(exact_cell_averages(config, ss.fv_t)[1:])[var]

# Selections: default near the discontinuity; reset there when the number
# of cells shrinks below the selected index (not to the boundary, where the
# Riemann problem is empty).
if kept("fv_interface", n_cells + 1) > n_cells:
    ss.keep_fv_interface = default_interface(UL, UR)
if kept("fv_cell", n_cells) > n_cells - 1:
    ss.keep_fv_cell = min(default_interface(UL, UR), n_cells - 1)

if ts.cfl < cfl:
    st.caption(
        f"CFL effectif {ts.cfl:g} au lieu de {cfl:g} : MUSCL, ENO et WENO sont plafonnés "
        f"à 0,5, au-delà duquel leur stabilité n'est plus garantie (§6.1)."
    )


def what_to_see(text: str) -> None:
    st.markdown("**Ce qu'il faut voir.** " + text)


def try_this(text: str) -> None:
    st.info("**À essayer.** " + text)


step = ss.fv_step
st.subheader(f"Étape {step} : {STEPS[step - 1]}")

if step == 1:
    st.plotly_chart(fig_averages(x, dx, U, var, gamma, exact), width="stretch")
    what_to_see(
        "L'inconnue du schéma n'est pas une valeur en un point, mais la **moyenne** de la "
        "solution sur chaque maille : la hauteur du palier, dont l'aire ombrée est la "
        "quantité contenue dans la maille. Ce sont les variables conservatives (ρ, ρu, E) "
        "qui sont moyennées ; ρ, u et p affichés en sont déduits (survolez une maille). "
        "L'aire ombrée n'est tracée que pour ρ : u et p ne sont pas des quantités "
        "conservées, leurs paliers sont calculés à partir des moyennes. "
        "À t = 0, avec un nombre pair de mailles, la discontinuité tombe sur une face et "
        "les paliers sont exacts (avec un nombre impair, la maille centrale mélange les "
        "deux états) ; après quelques pas, ils s'étalent."
    )
    try_this("passer de 8 à 32 mailles : les paliers épousent de mieux en mieux la "
             "solution. Afficher u : il est nul partout à t = 0.")
    st.markdown(doc_link("03_finite_volume.md", "Chapitre 3, §3.1 et §3.3"))

elif step == 2:
    flat = flat_cells(U, scheme, gamma, dx, var)
    st.plotly_chart(fig_reconstruction(x, dx, U, UL, UR, flat, var, gamma, recon == "muscl",
                                       exact), width="stretch")
    what_to_see(
        "Pour calculer le flux à une interface, il faut un état de chaque côté : U_L, venu "
        "de la maille de gauche, et U_R, venu de celle de droite. À l'ordre 1, ce sont "
        "simplement les moyennes voisines. À l'ordre élevé, on extrapole la solution "
        "jusqu'aux faces : dans les zones régulières, U_L et U_R se rapprochent et le saut "
        "(barres du bas) diminue, donc la dissipation du flux aussi. Le limiteur annule ou "
        "réduit fortement la pente là où les écarts voisins sont très différents : "
        "extremum, discontinuité, ou coin du profil comme le pied d'une détente (croix). "
        "Pour ENO et WENO, le code ne "
        "calcule que les valeurs aux faces : le pointillé les relie, ce n'est pas le "
        "polynôme. La reconstruction porte ici sur les variables primitives (§5.6)."
    )
    try_this("comparer Constante, MUSCL avec minmod puis avec superbee : regarder les "
             "barres de saut. Passer à WENO5 et observer les faces voisines de la "
             "discontinuité.")
    st.markdown(doc_link("05_reconstruction.md", "Chapitre 5 (§5.1 à §5.6)") + " · "
                + doc_link("03_finite_volume.md", "§3.4"))

elif step == 3:
    k = st.slider("Interface étudiée", 0, n_cells, ss.keep_fv_interface, key="fv_interface",
                  on_change=save, args=("fv_interface",))
    WL = tuple(float(v) for v in conservative_to_primitive(UL[:, k], gamma))
    WR = tuple(float(v) for v in conservative_to_primitive(UR[:, k], gamma))
    model = model_waves(flux, UL[:, k], UR[:, k], config.gas)
    st.plotly_chart(fig_riemann(WL, WR, exact_waves(WL, WR, gamma), model, dx, ts.dt, var),
                    width="stretch")
    if model is None:
        st.warning("AUSM ne résout pas de problème de Riemann : il sépare convection et "
                   "pression (§4.9). Seule la solution exacte est tracée.")
    if flux in ("roe", "roe-nc"):
        st.caption("Pour Roe, les lignes du modèle sont les valeurs propres u − a, u, u + a. "
                   "La correction d'entropie modifie |λ| dans la dissipation, pas ces vitesses.")
    what_to_see(
        "Chaque interface porte un petit **problème de Riemann** entre U_L et U_R. Sa "
        "solution exacte (traits noirs : choc, contact en tirets, éventail de détente) ne "
        "dépend que de x/t. Le flux est évalué sur la droite x/t = 0. Godunov y prend "
        "l'état exact ; les solveurs approchés remplacent l'éventail par un modèle plus "
        "simple (pointillés rouges). HLL n'a pas d'onde de contact. Le rectangle a la "
        "largeur d'une maille de chaque côté et la hauteur d'un pas de temps : les ondes "
        "ne doivent pas atteindre l'interface voisine pendant un pas (étape 6). Au "
        "démarrage de Sod, le choc exact sort pourtant du rectangle : le pas de temps est "
        "calculé avec max(|u| + c) des mailles (1,18), plus petit que la vitesse du choc "
        "qui va se former (1,75). C'est pourquoi on réduit souvent le CFL des premiers pas."
    )
    try_this("passer de HLL à HLLC, puis cliquer « +10 pas » : la ligne du contact "
             "apparaît. Comparer avec Rusanov (éventail ±S_max) après quelques pas : à t = 0 "
             "sur Sod, les vitesses de Davis sont symétriques et HLL coïncide avec Rusanov. "
             "Choisir une interface en zone régulière : les ondes sont faibles.")
    st.markdown(doc_link("02_riemann_problem.md", "Chapitre 2") + " · "
                + doc_link("04_flux_schemes.md", "Chapitre 4 (§4.1 à §4.5)"))

elif step == 4:
    component = COMPONENT_NAMES.index(st.radio(
        "Composante du flux", COMPONENT_NAMES, horizontal=True,
        index=COMPONENT_NAMES.index(kept("fv_component", COMPONENT_NAMES[0])),
        key="fv_component", on_change=save, args=("fv_component",)))
    k = ss.keep_fv_interface
    breakdown = flux_breakdown(UL, UR, scheme, config.gas, dx, ts.dt)
    st.plotly_chart(fig_fluxes(x, dx, U, breakdown, component, var, gamma, k),
                    width="stretch")
    what_to_see(
        "Chaque flèche est le flux numérique à une interface (sens et intensité) : ce qui "
        "sort d'une maille entre exactement dans la voisine, c'est la **conservation**. "
        "Le flux est consistant : si U_L = U_R = U, il vaut le flux physique F(U). À droite, "
        "à l'interface de l'étape 3, il se lit comme une moyenne centrée moins une "
        "**dissipation** proportionnelle au saut. Les flux décentrés diffèrent surtout par "
        "la taille de cette dissipation, la plus forte étant celle de Rusanov."
    )
    try_this("cliquer « +10 pas », choisir l'interface du contact à l'étape 3, puis "
             "comparer Rusanov, HLL et HLLC : la barre de dissipation diminue. Changer de "
             "flux ramène à t = 0 : avancer de nouveau avant de comparer (à t = 0 sur Sod, "
             "Rusanov et HLL donnent le même flux).")
    st.markdown(doc_link("03_finite_volume.md", "§3.2") + " · "
                + doc_link("04_flux_schemes.md", "§4.2 à §4.4"))

elif step == 5:
    F = flux_breakdown(UL, UR, scheme, config.gas, dx, ts.dt).numerical
    i = st.slider("Maille étudiée", 0, n_cells - 1, ss.keep_fv_cell, key="fv_cell",
                  on_change=save, args=("fv_cell",))
    U_new = stage_update(U, F, ts.dt, dx)
    st.plotly_chart(fig_update(x, dx, U, U_new, F, i, var, gamma), width="stretch")
    st.latex(r"\bar U_i^{+} = \bar U_i - \frac{\Delta t}{\Delta x}"
             r"\left(\hat F_{i+1/2} - \hat F_{i-1/2}\right)")
    st.dataframe(pd.DataFrame({
        "composante": ["ρ", "ρu", "E"],
        "Ū_i avant": U[:, i], "F̂ entrant (gauche)": F[:, i],
        "F̂ sortant (droite)": F[:, i + 1], "Ū_i après": U_new[:, i],
    }), hide_index=True, width="stretch")
    b = global_balance(U, U_new, F, ts.dt, dx)
    st.dataframe(pd.DataFrame({
        "total": ["masse", "quantité de mouvement", "énergie"],
        "variation sur le domaine": b.change,
        "Δt × (flux bord gauche − flux bord droit)": b.boundary,
    }), hide_index=True, width="stretch")
    what_to_see(
        "La moyenne d'une maille change de la différence entre ce qui entre et ce qui "
        "sort. Sommée sur le domaine, cette différence se simplifie d'une interface à "
        "l'autre : la variation des totaux vient **uniquement des bords**, les deux "
        "colonnes du second tableau sont égales. Sur Sod, tant que les ondes n'ont pas "
        "atteint les bords, la masse totale ne change pas (u = 0 aux bords) et la quantité "
        "de mouvement change, car les pressions 1 et 0,1 des bords agissent comme des "
        "forces. « Conservatif » ne veut pas dire que rien "
        "ne change."
    )
    try_this("choisir une maille loin de la discontinuité : ses moyennes ne bougent pas.")
    st.markdown(doc_link("03_finite_volume.md", "§3.3"))

else:
    stages = rk_stages(U, scheme, config, rk, ts.dt)
    rk_used = select_time_integrator(scheme, rk)
    st.plotly_chart(fig_stages(x, dx, stages, var, gamma, ts.i_max), width="stretch")
    st.markdown(
        f"**{len(stages) - 1} évaluation(s) de l'opérateur spatial par pas avec {rk_used}** : "
        f"chaque étage refait les étapes 2 à 5 (reconstruction, problèmes de Riemann, flux, "
        f"bilan) sur l'état de l'étage."
    )
    st.latex(
        rf"\Delta t = \mathrm{{CFL}}\,\frac{{\Delta x}}{{\max(|u| + c)}}"
        rf" = {ts.cfl:g} \times \frac{{{dx:.4f}}}{{{ts.s_max:.4f}}} = {ts.dt:.5f}"
    )
    what_to_see(
        "La méthode des lignes sépare l'espace (étapes 2 à 5, qui donnent dU/dt) et le "
        "temps, intégré par une méthode de Runge-Kutta. Le pas de temps est fixé par "
        "l'onde la plus rapide (maille colorée) : elle ne doit pas traverser plus d'une "
        "fraction de maille par pas, comme le montrait le rectangle de l'étape 3. Les "
        "reconstructions sont plafonnées à CFL 0,5 : au-delà, MUSCL n'est plus TVD, et "
        "WENO, qui n'est pas TVD, reçoit le même plafond."
    )
    try_this("avec MUSCL, passer de RK1 à RK2 et comparer Uⁿ⁺¹. Avec la reconstruction "
             "constante, monter le CFL jusqu'à 1. La stabilité au-delà s'étudie sur la page "
             "Analyse de Fourier.")
    st.markdown(doc_link("06_time_integration.md", "Chapitre 6 (§6.1, §6.4, §6.7)"))
