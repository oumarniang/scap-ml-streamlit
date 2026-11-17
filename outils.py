from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


RACINE = Path(__file__).resolve().parent


@st.cache_resource
def charger_modele(nom):
    chemin = RACINE / "modeles" / f"{nom}.joblib"

    if not chemin.exists():
        st.error(
            f"Modèle introuvable : {chemin.name}. "
            "Lancez d'abord python entrainer.py."
        )
        st.stop()

    return joblib.load(chemin)


@st.cache_data
def charger_donnees(nom, **options):
    return pd.read_csv(
        RACINE / "donnees" / nom,
        sep=";",
        **options
    )


def fr(valeur, unite="", decimales=0):
    texte = (
        f"{valeur:,.{decimales}f}"
        .replace(",", " ")
        .replace(".", ",")
    )

    return f"{texte} {unite}".strip()