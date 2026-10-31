"""Entry point of the Streamlit app: navigation between the three pages.

Streamlit Cloud runs this file. The home page explains the finite-volume
method step by step; the solver and the Fourier analysis are the pages
where students experiment.
"""

import streamlit as st

pages = st.navigation([
    st.Page("pages/finite_volumes.py", title="Comprendre les volumes finis", default=True),
    st.Page("pages/solver.py", title="Solveur"),
    st.Page("pages/2_Fourier_Analysis.py", title="Analyse de Fourier"),
])
pages.run()
