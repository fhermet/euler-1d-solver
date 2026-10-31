"""Home page: the finite-volume method, step by step (in French)."""

import streamlit as st

st.set_page_config(page_title="Comprendre les volumes finis", layout="wide")
st.title("Comprendre la méthode des volumes finis")
st.markdown(
    "Cette page suit un pas de temps du solveur, maille par maille, sur le tube de Sod."
)
