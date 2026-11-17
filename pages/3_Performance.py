import pandas as pd
import streamlit as st

from outils import charger_modele, fr


st.title("📈 Que valent les modèles ?")

auto = charger_modele("modele_auto")
audition = charger_modele("modele_audition")

st.write(
    "Ces mesures proviennent des modèles d'évaluation, "
    "testés sur les 20 % de données mis de côté "
    "pendant l'entraînement."
)

lineaire, logistique = st.tabs([
    "Prix voiture",
    "Test auditif"
])


# ---------------------------------------------------------
# RÉGRESSION LINÉAIRE
# ---------------------------------------------------------

with lineaire:
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "R² en test",
        f"{auto['r2']:.2f}".replace(".", ",")
    )

    c2.metric(
        "Erreur moyenne (MAE)",
        fr(auto["mae"], "$")
    )

    c3.metric(
        "Voitures du jeu de test",
        len(auto["test"])
    )

    st.subheader("Prix réel et prix prédit")

    st.scatter_chart(
        auto["test"],
        x="reel",
        y="prevu"
    )

    st.caption(
        "Chaque point représente une voiture du jeu de test. "
        "Une prédiction parfaite aurait un prix prédit "
        "égal au prix réel."
    )

    st.subheader("Coefficients du modèle final")

    st.bar_chart(auto["coefficients"].sort_values())

    st.caption(
        "Les variables numériques ont été standardisées. "
        "Les coefficients décrivent le modèle ; "
        "ils ne prouvent pas une relation de cause à effet. "
        "Les catégories rares doivent être interprétées "
        "avec prudence."
    )


# ---------------------------------------------------------
# RÉGRESSION LOGISTIQUE
# ---------------------------------------------------------

with logistique:
    c1, c2 = st.columns(2)

    c1.metric(
        "Exactitude en test",
        f"{audition['exactitude']:.1%}".replace(".", ",")
    )

    c2.metric(
        "AUC en test",
        f"{audition['auc']:.2f}".replace(".", ",")
    )

    st.subheader("Matrice de confusion — seuil de 0,50")

    matrice = pd.DataFrame(
        audition["matrice"],
        index=["Réel : échec", "Réel : réussite"],
        columns=["Prédit : échec", "Prédit : réussite"]
    )

    st.dataframe(matrice, use_container_width=True)

    st.caption(
        "Les lignes indiquent les résultats réels "
        "et les colonnes les prédictions."
    )

    st.info(
        f"Dans le fichier complet, "
        f"{audition['taux_reussite']:.0%} des personnes "
        "réussissent le test. Prédire systématiquement "
        "« réussite » donnerait donc déjà ce taux "
        "d'exactitude sur ce fichier."
    )