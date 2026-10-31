"""Home page: the finite-volume method in eight short screens.

Screens 1 to 6 advect a square wave (euler1d.advection) to show the whole
mechanism; screens 7 and 8 come back to the Euler equations on Sod. Each
screen asks one question and shows one graph with one bubble.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import streamlit as st

from euler1d.advection import PHI, SQUARE, advect, exact_averages
from euler1d.fv_walkthrough import contact_cells, sod_setup
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from ui_common import documentation_sidebar
from ui_walkthrough import (
    CFL_SCREEN_4, CFL_SCREEN_6, SLOPE_LABELS, fig_screen_1, fig_screen_2, fig_screen_3,
    fig_screen_4, fig_screen_5, fig_screen_6, fig_screen_7, fig_screen_8,
)

st.set_page_config(page_title="Comprendre les volumes finis", layout="wide")

with st.sidebar:
    documentation_sidebar()

st.title("Comprendre la méthode des volumes finis")

N_CELLS = 40
CFL_SCREEN_3 = 0.8
SOD_CELLS, SOD_T = 100, 0.2
SHORT_TITLES = {1: "Le problème", 2: "Les mailles", 3: "Le bilan", 4: "Le flux",
                5: "Le pas de temps", 6: "Monter en ordre", 7: "Trois ondes", 8: "Comparer"}
QUESTIONS = {
    1: "Que doit calculer le schéma ?",
    2: "Que garde-t-on de la solution ?",
    3: "Comment une maille change-t-elle ?",
    4: "Quelle valeur prendre à une face ?",
    5: "De combien peut-on avancer à chaque pas ?",
    6: "Comment garder un front raide ?",
    7: "Pourquoi l'advection ne suffit-elle plus ?",
    8: "Qu'est-ce que ça change sur la solution ?",
}
U0 = exact_averages(N_CELLS, 0.0)

ss = st.session_state


def kept(key: str, default):
    """Current value of the widget *key*, kept in a key of its own.

    Streamlit deletes a widget's key when the widget is not drawn in a run
    (another screen, another page): each widget reads its value from
    'keep_<key>' and writes it back there when the student changes it.
    """
    return ss.setdefault("keep_" + key, default)


def save(key: str) -> None:
    ss["keep_" + key] = ss[key]


def slider(label, lo, hi, default, key, step=None):
    return st.slider(label, lo, hi, kept(key, default), step=step, key=key, on_change=save,
                     args=(key,))


def toggle(label, options: dict, default, key):
    """Horizontal radio over the keys of *options*, labelled by its values."""
    choices = list(options)
    return st.radio(label, choices, index=choices.index(kept(key, default)), horizontal=True,
                    format_func=options.get, key=key, on_change=save, args=(key,))


# ---------------------------------------------------------------------------
# Common template
# ---------------------------------------------------------------------------

def _goto(screen: int) -> None:
    ss.fv_screen = min(max(screen, 1), len(QUESTIONS))


def header(n: int) -> None:
    """Progress bar (chapter Advection 1-6, chapter Euler 7-8), then the question."""
    cols = st.columns([1] * 6 + [0.8] + [1] * 2)
    for s in range(1, 9):
        col = cols[s - 1] if s <= 6 else cols[s]
        col.button(str(s), key=f"fv_goto_{s}", on_click=_goto, args=(s,),
                   type="primary" if s == n else "secondary", width="stretch",
                   help=SHORT_TITLES[s])
    cols[6].markdown("<div style='text-align:center;padding-top:0.4rem'><b>Euler :</b></div>",
                     unsafe_allow_html=True)
    chapter = "Advection" if n <= 6 else "Euler"
    st.caption(f"{chapter} · écran {n} sur 8 · {SHORT_TITLES[n]}")
    st.header(QUESTIONS[n])


def show(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def explain(text: str) -> None:
    st.markdown(text)


def try_this(text: str) -> None:
    st.info("**À essayer :** " + text)


def nav(n: int) -> None:
    left, _, right = st.columns([2, 3, 2])
    if n > 1:
        left.button(f"◀ {n - 1}. {SHORT_TITLES[n - 1]}", key="fv_prev", on_click=_goto,
                    args=(n - 1,), width="stretch")
    if n < 8:
        right.button(f"{n + 1}. {SHORT_TITLES[n + 1]} ▶", key="fv_next", on_click=_goto,
                     args=(n + 1,), width="stretch")


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def screen_1() -> None:
    st.markdown("On transporte un créneau : $u_t + a\\,u_x = 0$, avec $a = 1$, "
                "sur $[0, 1]$ périodique.")
    t = slider("Temps t", 0.0, 1.0, 0.0, "fv_t1", step=0.05)
    show(fig_screen_1(t))
    explain("L'équation transporte le profil à la vitesse a. C'est ce que le schéma doit "
            "reproduire.")
    try_this("faites avancer t.")


def screen_2() -> None:
    n = slider("Nombre de mailles N", 10, 80, 10, "fv_n2")
    show(fig_screen_2(n))
    explain("On découpe en mailles et on ne garde que la moyenne de chacune. Une maille "
            "coupée par le front prend une valeur intermédiaire.")
    try_this("passez de 10 à 80 mailles.")


def _step3() -> None:
    ss.fv_steps3 = ss.get("fv_steps3", 0) + 1


def _reset3() -> None:
    ss.fv_steps3 = 0


def screen_3() -> None:
    ss.setdefault("fv_steps3", 0)
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.button("Un pas", key="fv_step3", on_click=_step3, type="primary", width="stretch")
    c2.button("Revenir à t = 0", key="fv_reset3", on_click=_reset3, width="stretch")
    n = ss.fv_steps3
    c3.markdown(f"Pas n° {n + 1} · décentré amont, CFL {CFL_SCREEN_3:g}".replace(".", ","))
    cell = int(SQUARE[1] * N_CELLS)   # the cell holding the front edge at t = 0
    before = advect(U0, "upwind", CFL_SCREEN_3, n).u
    after = advect(before, "upwind", CFL_SCREEN_3, 1).u
    show(fig_screen_3(before, after, cell, CFL_SCREEN_3))
    explain("La moyenne change de ce qui entre moins ce qui sort, pendant Δt. Ce qui sort "
            "d'une maille entre dans la voisine.")
    try_this("faites un pas, puis un autre : le niveau de la maille rose monte.")


def screen_4() -> None:
    c1, c2 = st.columns(2)
    with c1:
        scheme = toggle("Valeur à la face", {"upwind": "décentré amont", "centred": "centré"},
                        "upwind", "fv_scheme4")
    with c2:
        n = slider("Nombre de pas", 0, 40, 10, "fv_n4")
    show(fig_screen_4(advect(U0, scheme, CFL_SCREEN_4, n), scheme, n))
    explain("On ne connaît que les deux moyennes voisines d'une face. Le décentré prend la "
            "maille d'où vient l'onde ; le centré prend leur moyenne.")
    try_this("passez au centré.")


def screen_5() -> None:
    c1, c2 = st.columns(2)
    with c1:
        cfl = slider("Nombre de CFL", 0.2, 1.5, 0.5, "fv_cfl5", step=0.1)
    with c2:
        n = slider("Nombre de pas", 0, 40, 20, "fv_n5")
    show(fig_screen_5(advect(U0, "upwind", cfl, n), cfl, n))
    explain("Pendant un pas, l'onde parcourt CFL mailles. Au-delà d'une maille, le schéma va "
            "chercher une information qu'il n'a pas.")
    try_this("réglez le CFL à 1, puis à 1,1.")


def screen_6() -> None:
    c1, c2 = st.columns([3, 2])
    with c1:
        slope = toggle("Valeur à la face", SLOPE_LABELS, "order1", "fv_slope6")
    with c2:
        t = slider("Temps t", 0.0, 1.0, 0.5, "fv_t6", step=0.05)
    n_steps = round(t * N_CELLS / CFL_SCREEN_6)
    if slope == "order1":
        u = advect(U0, "upwind", CFL_SCREEN_6, n_steps).u
    else:
        u = advect(U0, "muscl", CFL_SCREEN_6, n_steps, PHI[slope]).u
    show(fig_screen_6(u, slope, t))
    explain("Une pente dans chaque maille rapproche les valeurs de part et d'autre des faces. "
            "Le limiteur l'annule là où elle créerait une oscillation.")
    try_this("comparez les quatre.")


def screen_7() -> None:
    choice = toggle("Ondes affichées", {"exact": "Exact", "hll": "HLL", "hllc": "HLLC"},
                    "hll", "fv_choice7")
    show(fig_screen_7(choice))
    explain("En advection, une seule onde quitte la face, dans un seul sens. Pour Euler, trois "
            "ondes partent de chaque face, dans les deux sens : le solveur de Riemann dit ce "
            "qui traverse la face.")
    try_this("passez de HLL à HLLC.")


@st.cache_data(show_spinner=False)
def _sod_density(flux: str) -> np.ndarray:
    cfg = sod_setup(SOD_CELLS, 0.9)
    cfg = replace(cfg, time=replace(cfg.time, t_final=SOD_T))
    return run_simulation(cfg, get_scheme(flux), n_snapshots=0).snapshots[-1][0]


def screen_8() -> None:
    flux = toggle("Solveur de Riemann", {"rusanov": "Rusanov", "hll": "HLL", "hllc": "HLLC"},
                  "hll", "fv_flux8")
    x = (np.arange(SOD_CELLS) + 0.5) / SOD_CELLS
    rho = _sod_density(flux)
    show(fig_screen_8(x, rho, flux, contact_cells(rho, x, SOD_T)))
    explain("Plus le modèle d'ondes est fidèle, moins le contact est étalé. Pour aller plus "
            "loin, la page **Solveur** compare tous les schémas.")
    try_this("comparez les trois.")


SCREENS = {1: screen_1, 2: screen_2, 3: screen_3, 4: screen_4, 5: screen_5, 6: screen_6,
           7: screen_7, 8: screen_8}

ss.setdefault("fv_screen", 1)
screen = ss.fv_screen
header(screen)
SCREENS[screen]()
nav(screen)
