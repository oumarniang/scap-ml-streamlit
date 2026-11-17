import numpy as np
import pandas as pd
import streamlit as st

from outils import charger_modele


st.title("🎧 Réussira-t-il le test auditif ?")
st.caption("Démonstration pédagogique sur les données du tutoriel.")

paquet = charger_modele("modele_audition")
modele = paquet["modele"]

age = st.slider(
    "Âge",
    min_value=18,
    max_value=90,
    value=50
)

score = st.slider(
    "Score physique",
    min_value=0.0,
    max_value=50.0,
    value=32.0,
    step=0.5
)

seuil = st.sidebar.slider(
    "Seuil de décision",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Si la probabilité atteint ce seuil, on prédit une réussite."
)

# Préparer les caractéristiques du profil
profil = pd.DataFrame([{
    "age": age,
    "physical_score": score
}])

# Probabilité de la classe 1 : réussite
p = float(modele.predict_proba(profil)[0, 1])

c1, c2 = st.columns(2)

c1.metric("Probabilité de réussite", f"{p:.1%}".replace(".", ","))

c2.metric(
    "Décision",
    "Réussite" if p >= seuil else "Échec"
)

st.progress(p)

# Faire varier l'âge en gardant le même score physique
st.subheader("Comment la probabilité évolue avec l'âge")

ages = np.arange(18, 91)

profils = pd.DataFrame({
    "age": ages,
    "physical_score": score
})

courbe = pd.DataFrame({
    "Âge": ages,
    "Probabilité de réussite": modele.predict_proba(profils)[:, 1]
}).set_index("Âge")

st.line_chart(courbe)

st.caption(
    f"Courbe calculée pour un score physique constant de {score:.1f}. "
    "Le seuil modifie la décision, sans modifier la probabilité."
)