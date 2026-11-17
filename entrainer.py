from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RACINE = Path(__file__).resolve().parent
(RACINE / "modeles").mkdir(exist_ok=True)


# =========================================================
# 1. PRIX D'UNE VOITURE : RÉGRESSION LINÉAIRE
# =========================================================

autos = pd.read_csv(
    RACINE / "donnees" / "automobile_data.csv",
    sep=";",
    na_values="?"
)

NUM = [
    "wheel-base", "length", "width", "curb-weight",
    "engine-size", "horsepower", "city-mpg", "highway-mpg"
]

CAT = ["body-style", "drive-wheels", "fuel-system"]


def construire_pipeline():
    preparation = ColumnTransformer([
        (
            "num",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler()
            ),
            NUM
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CAT
        )
    ])

    return Pipeline([
        ("prep", preparation),
        ("modele", LinearRegression())
    ])


# Garder 20 % des données pour évaluer le modèle
X_tr, X_te, y_tr, y_te = train_test_split(
    autos[NUM + CAT],
    autos["price"],
    test_size=0.2,
    random_state=42
)

essai = construire_pipeline().fit(X_tr, y_tr)
prevu = essai.predict(X_te)

mae = mean_absolute_error(y_te, prevu)
r2 = r2_score(y_te, prevu)

print(f"Prix voiture : R² test = {r2:.3f} ; MAE = {mae:,.0f} $")

# Réentraîner le modèle livré sur toutes les données
final = construire_pipeline().fit(
    autos[NUM + CAT],
    autos["price"]
)

noms = final.named_steps["prep"].get_feature_names_out()

coefficients = pd.Series(
    final.named_steps["modele"].coef_,
    index=[nom.split("__", 1)[1] for nom in noms]
)

joblib.dump(
    {
        "modele": final,
        "colonnes": NUM + CAT,
        "mae": mae,
        "r2": r2,
        "n": len(autos),
        "categories": {
            colonne: sorted(
                autos[colonne].dropna().unique().tolist()
            )
            for colonne in CAT
        },
        "test": pd.DataFrame({
            "reel": y_te.values,
            "prevu": prevu
        }),
        "coefficients": coefficients,
        "sklearn": sklearn.__version__,
    },
    RACINE / "modeles" / "modele_auto.joblib"
)


# =========================================================
# 2. TEST AUDITIF : RÉGRESSION LOGISTIQUE
# =========================================================

audition = pd.read_csv(
    RACINE / "donnees" / "hearing_test.csv",
    sep=";"
)

COL = ["age", "physical_score"]

X_tr, X_te, y_tr, y_te = train_test_split(
    audition[COL],
    audition["test_result"],
    test_size=0.2,
    random_state=42,
    stratify=audition["test_result"]
)

essai = make_pipeline(
    StandardScaler(),
    LogisticRegression()
).fit(X_tr, y_tr)

proba = essai.predict_proba(X_te)[:, 1]
predictions = proba >= 0.5

exactitude = accuracy_score(y_te, predictions)
auc = roc_auc_score(y_te, proba)

print(
    f"Test auditif : exactitude = {exactitude:.3f} ; "
    f"AUC = {auc:.3f}"
)

# Réentraîner le modèle livré sur toutes les données
final = make_pipeline(
    StandardScaler(),
    LogisticRegression()
).fit(audition[COL], audition["test_result"])

joblib.dump(
    {
        "modele": final,
        "colonnes": COL,
        "exactitude": exactitude,
        "auc": auc,
        "n": len(audition),
        "matrice": confusion_matrix(y_te, predictions),
        "taux_reussite": float(
            audition["test_result"].mean()
        ),
        "sklearn": sklearn.__version__,
    },
    RACINE / "modeles" / "modele_audition.joblib"
)

print("Entraînement terminé.")
print("Modèles enregistrés dans :", RACINE / "modeles")
print("Version de scikit-learn :", sklearn.__version__)