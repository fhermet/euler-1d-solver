"""Entry point of the Streamlit app: navigation between the pages.

Streamlit Cloud runs this file. The solver and the Fourier analysis are the
pages where students experiment. The page explaining the finite-volume
method is work in progress: it is shown, as the home page, only when the
environment variable SHOW_WALKTHROUGH is set
(SHOW_WALKTHROUGH=1 streamlit run Solveur_Euler_1D.py).
"""

import os

import streamlit as st

show_walkthrough = bool(os.environ.get("SHOW_WALKTHROUGH"))
pages = [
    st.Page("pages/solver.py", title="Solveur", default=not show_walkthrough),
    st.Page("pages/2_Fourier_Analysis.py", title="Analyse de Fourier"),
]
if show_walkthrough:
    pages.insert(0, st.Page("pages/finite_volumes.py", title="Comprendre les volumes finis",
                            default=True))
st.navigation(pages).run()
