import streamlit as st

st.set_page_config(
    page_title="Mes modèles de ML",
    page_icon="🤖",
    layout="wide"
)

from outils import charger_modele, fr


auto = charger_modele("modele_auto")
audition = charger_modele("modele_audition")

st.title("🤖 Mes modèles de machine learning")

st.write(
    "Cette application présente deux modèles "
    "entraînés sur les données du tutoriel."
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("🚗 Prix d'une voiture")

    st.write(
        "**Régression linéaire** : estime un prix "
        "à partir des caractéristiques d'une voiture."
    )

    st.metric(
        "Erreur moyenne en test",
        fr(auto["mae"], "$")
    )

with c2:
    st.subheader("🎧 Test auditif")

    st.write(
        "**Régression logistique** : estime la probabilité "
        "de réussir le test à partir de l'âge "
        "et d'un score physique."
    )

    st.metric(
        "Exactitude en test",
        f"{audition['exactitude']:.1%}".replace(".", ",")
    )

st.divider()

st.caption(
    "SCAP · Machine learning · Oumar Niang · "
    "Données d'illustration — démonstration pédagogique."
)