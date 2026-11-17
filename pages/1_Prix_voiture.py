import pandas as pd
import streamlit as st

from outils import charger_modele, fr


st.title("🚗 Estimer le prix d'une voiture")

paquet = charger_modele("modele_auto")
modele = paquet["modele"]
mae = paquet["mae"]

st.caption(
    "Données historiques : prix exprimés en dollars de 1985."
)

une, plusieurs = st.tabs([
    "Une voiture",
    "Un fichier de voitures"
])


# ---------------------------------------------------------
# ONGLET 1 : ESTIMER UNE VOITURE
# ---------------------------------------------------------

with une:
    with st.form("voiture"):
        c1, c2, c3 = st.columns(3)

        carrosserie = c1.selectbox(
            "Carrosserie",
            paquet["categories"]["body-style"]
        )

        transmission = c1.selectbox(
            "Roues motrices",
            paquet["categories"]["drive-wheels"]
        )

        injection = c1.selectbox(
            "Système d'injection",
            paquet["categories"]["fuel-system"]
        )

        empattement = c2.number_input(
            "Empattement (pouces)", 85.0, 125.0, 98.0
        )

        longueur = c2.number_input(
            "Longueur (pouces)", 140.0, 210.0, 172.0
        )

        largeur = c2.number_input(
            "Largeur (pouces)", 60.0, 73.0, 66.0
        )

        poids = c2.number_input(
            "Poids à vide (livres)", 1400, 4200, 2500
        )

        moteur = c3.number_input(
            "Cylindrée (pouces cubes)", 60, 330, 120
        )

        puissance = c3.number_input(
            "Puissance (ch)", 45, 290, 100
        )

        ville = c3.number_input(
            "Consommation ville (mpg)", 13, 50, 25
        )

        route = c3.number_input(
            "Consommation route (mpg)", 16, 55, 30
        )

        envoye = st.form_submit_button("Estimer")

    if envoye:
        voiture = pd.DataFrame([{
            "wheel-base": empattement,
            "length": longueur,
            "width": largeur,
            "curb-weight": poids,
            "engine-size": moteur,
            "horsepower": puissance,
            "city-mpg": ville,
            "highway-mpg": route,
            "body-style": carrosserie,
            "drive-wheels": transmission,
            "fuel-system": injection
        }])

        prix = float(modele.predict(voiture)[0])

        st.metric("Prix estimé", fr(prix, "$"))

        st.caption(
            f"Erreur moyenne mesurée en test : {fr(mae, '$')}."
        )

        if prix >= 0:
            st.write(
                f"Ordre de grandeur : "
                f"{fr(max(prix - mae, 0), '$')} à "
                f"{fr(prix + mae, '$')}."
            )
            st.caption(
                "Cette fourchette correspond au prix estimé "
                "± l'erreur moyenne. Ce n'est pas un intervalle "
                "de confiance statistique."
            )
        else:
            st.warning(
                "Le modèle produit un prix négatif : "
                "cette estimation n'est pas exploitable."
            )


# ---------------------------------------------------------
# ONGLET 2 : ESTIMER PLUSIEURS VOITURES
# ---------------------------------------------------------

with plusieurs:
    fichier = st.file_uploader(
        "Importer un CSV avec les mêmes colonnes "
        "que le jeu d'entraînement (séparateur ;)",
        type=["csv"]
    )

    if fichier is not None:
        try:
            voitures = pd.read_csv(
                fichier,
                sep=";",
                na_values="?"
            )

            manquantes = [
                colonne
                for colonne in paquet["colonnes"]
                if colonne not in voitures.columns
            ]

            if manquantes:
                st.error(
                    "Colonnes absentes : "
                    + ", ".join(manquantes)
                )

            elif voitures.empty:
                st.warning("Le fichier ne contient aucune voiture.")

            else:
                voitures["prix estimé"] = modele.predict(
                    voitures[paquet["colonnes"]]
                ).round(0)

                st.dataframe(
                    voitures,
                    use_container_width=True
                )

                st.download_button(
                    "Télécharger les estimations",
                    data=voitures.to_csv(
                        sep=";",
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="estimations.csv",
                    mime="text/csv"
                )

        except (ValueError, UnicodeError, pd.errors.EmptyDataError) as erreur:
            st.error(
                f"Impossible de traiter ce fichier : {erreur}"
            )