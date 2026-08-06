import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent



def get_preprocessing_values(X_train):
    """
    Calcule les informations nécessaires pour traiter
    les nouveaux passagers.

    Ces informations sont calculées UNIQUEMENT sur le train.
    """

    preprocessing_values = {

        # Age :
        # si Age est None pour un nouveau passager,
        # on utilisera cette médiane.
        "Age_median": X_train["Age"].median(),

        # Embarked :
        # si Embarked est None,
        # on utilisera cette modalité.
        "Embarked_mode": X_train["Embarked"].mode()[0]
    }

    return preprocessing_values

def save_preprocessing_values(
    preprocessing_values,
    path="models/preprocessing_values.pkl"
):
    """
    Sauvegarde les informations calculées sur le train.
    """

    joblib.dump(
        preprocessing_values,
        path
    )

import pandas as pd
import numpy as np



def prepare_new_data_for_prediction(
    X,
    scaler
):


    preprocessing_info = joblib.load(
    BASE_DIR / "models" / "preprocessing_info.pkl"

)

    training_medians = preprocessing_info["training_medians"]

    training_modes = preprocessing_info["training_modes"]
    # ============================================================
    # 0. COPIE ET PARAMÈTRES
    # ============================================================

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "ERREUR : X doit être un DataFrame pandas."
        )

    if X.empty:
        raise ValueError(
            "ERREUR : le DataFrame fourni est vide."
        )

    X = X.copy()

    # Uniformisation de None en NaN
    X = X.replace({None: np.nan})

    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Cabin_known",
        "FamilySize",
        "IsAlone",
        "Embarked_Q",
        "Embarked_S",
        "Title_Miss",
        "Title_Mr",
        "Title_Mrs",
        "Title_Rare"
    ]

    required_columns = [
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Cabin",
        "Embarked"
    ]

    print("=" * 70)
    print("ÉTAPE 0 — DONNÉES BRUTES")
    print("=" * 70)

    print("Shape :", X.shape)
    print("Colonnes :", X.columns.tolist())

    # Survived est la cible
    if "Survived" in X.columns:
        raise ValueError(
            "ERREUR : la colonne 'Survived' est la cible et ne doit "
            "pas être présente dans les données de prédiction."
        )

    # PassengerId est facultatif mais non utilisé
    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])

    # Vérification de la présence des colonnes brutes
    missing_columns = [
        col
        for col in required_columns
        if col not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"ERREUR : colonnes obligatoires absentes : "
            f"{missing_columns}. Les colonnes doivent exister, "
            "même si leurs valeurs sont None."
        )

    print("✓ Survived absent")
    print("✓ PassengerId supprimé si présent")
    print("✓ Toutes les colonnes brutes obligatoires sont présentes")


    # ============================================================
    # 1. GESTION DES VALEURS NUMÉRIQUES MANQUANTES
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 1 — IMPUTATION DES VARIABLES NUMÉRIQUES")
    print("=" * 70)

    numeric_columns = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    for col in numeric_columns:

        if col not in training_medians:
            raise ValueError(
                f"ERREUR : la médiane d'entraînement de '{col}' "
                "est absente de training_medians."
            )

        # Convertit les valeurs en numérique.
        # Une chaîne invalide devient NaN puis sera imputée.
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        missing_before = X[col].isna().sum()

        # Conserve les valeurs connues et remplace uniquement
        # les valeurs None/NaN.
        X[col] = X[col].fillna(
            training_medians[col]
        )

        if missing_before > 0:
            print(
                f"✓ {col} : {missing_before} valeur(s) remplacée(s) "
                f"par {training_medians[col]}"
            )
        else:
            print(f"✓ {col} : toutes les valeurs étaient renseignées")


    # ============================================================
    # 2. GESTION DES VARIABLES CATÉGORIELLES MANQUANTES
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 2 — IMPUTATION DES VARIABLES CATÉGORIELLES")
    print("=" * 70)

    categorical_columns = [
        "Sex",
        "Embarked"
    ]

    for col in categorical_columns:

        if col not in training_modes:
            raise ValueError(
                f"ERREUR : le mode d'entraînement de '{col}' "
                "est absent de training_modes."
            )

        # Les chaînes vides sont considérées comme manquantes
        X[col] = X[col].replace(
            r"^\s*$",
            np.nan,
            regex=True
        )

        missing_before = X[col].isna().sum()

        X[col] = X[col].fillna(
            training_modes[col]
        )

        if missing_before > 0:
            print(
                f"✓ {col} : {missing_before} valeur(s) remplacée(s) "
                f"par '{training_modes[col]}'"
            )
        else:
            print(f"✓ {col} : toutes les valeurs étaient renseignées")

    # Name peut être manquant :
    # dans ce cas le titre deviendra Rare.
    X["Name"] = X["Name"].fillna("")

    print("✓ Name manquant traité comme titre Rare")

    # Cabin reste volontairement manquante :
    # son absence sert à créer Cabin_known.
    print("✓ Cabin manquante conservée pour calculer Cabin_known")


    # ============================================================
    # 3. CRÉATION DES VARIABLES DÉRIVÉES
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 3 — CRÉATION DES VARIABLES")
    print("=" * 70)

    # Cabin_known :
    # 1 si la cabine est renseignée
    # 0 si elle vaut None, NaN ou chaîne vide
    cabin_text = (
        X["Cabin"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    X["Cabin_known"] = (
        ~cabin_text.isin(
            ["", "nan", "none"]
        )
    ).astype(int)

    # FamilySize
    X["FamilySize"] = (
        X["SibSp"]
        + X["Parch"]
        + 1
    )

    # IsAlone
    X["IsAlone"] = (
        X["FamilySize"] == 1
    ).astype(int)

    print("✓ Cabin_known créée")
    print("✓ FamilySize créée")
    print("✓ IsAlone créée")

    print("\nVérification :")

    print(
        X[
            [
                "Cabin_known",
                "FamilySize",
                "IsAlone"
            ]
        ]
    )


    # ============================================================
    # 4. CRÉATION DE TITLE
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 4 — CRÉATION DE TITLE")
    print("=" * 70)

    X["Title"] = (
        X["Name"]
        .astype(str)
        .str.extract(
            r",\s*([^.]*)\."
        )[0]
        .str.strip()
    )

    # Harmonisation des titres
    X["Title"] = X["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    common_titles = [
        "Miss",
        "Mr",
        "Mrs",
        "Master"
    ]

    # Tout titre différent, absent ou non reconnu devient Rare
    X["Title"] = X["Title"].where(
        X["Title"].isin(common_titles),
        "Rare"
    )

    print("Titres obtenus :")
    print(X["Title"].value_counts(dropna=False))

    valid_titles = [
        "Miss",
        "Mr",
        "Mrs",
        "Master",
        "Rare"
    ]

    if not X["Title"].isin(valid_titles).all():
        raise ValueError(
            "ERREUR : certains titres n'ont pas été correctement traités."
        )

    print("✓ Tous les titres appartiennent aux catégories attendues")


    # ============================================================
    # 5. ENCODAGE DE SEX
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 5 — ENCODAGE DE SEX")
    print("=" * 70)

    X["Sex"] = (
        X["Sex"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Une valeur différente de male/female est remplacée
    # par le mode d'entraînement.
    sex_mode = str(
        training_modes["Sex"]
    ).strip().lower()

    X["Sex"] = X["Sex"].where(
        X["Sex"].isin(
            ["female", "male"]
        ),
        sex_mode
    )

    X["Sex"] = X["Sex"].map({
        "female": 1,
        "male": 0
    })

    if X["Sex"].isna().any():
        raise ValueError(
            "ERREUR : certaines valeurs de Sex n'ont pas pu "
            "être encodées."
        )

    print("Valeurs Sex encodées :", X["Sex"].unique())
    print("✓ Sex correctement encodé : female=1, male=0")


    # ============================================================
    # 6. NORMALISATION DE EMBARKED
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 6 — NORMALISATION DE EMBARKED")
    print("=" * 70)

    X["Embarked"] = (
        X["Embarked"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    embarked_mode = str(
        training_modes["Embarked"]
    ).strip().upper()

    # Une valeur inconnue est remplacée par le mode du train
    X["Embarked"] = X["Embarked"].where(
        X["Embarked"].isin(
            ["C", "Q", "S"]
        ),
        embarked_mode
    )

    if not X["Embarked"].isin(["C", "Q", "S"]).all():
        raise ValueError(
            "ERREUR : certaines valeurs de Embarked "
            "n'ont pas pu être corrigées."
        )

    print("Valeurs Embarked :", X["Embarked"].unique())
    print("✓ Embarked correctement normalisé")


    # ============================================================
    # 7. ONE-HOT ENCODING
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 7 — ONE-HOT ENCODING")
    print("=" * 70)

    embarked_categories = [
        "C",
        "Q",
        "S"
    ]

    title_categories = [
        "Master",
        "Miss",
        "Mr",
        "Mrs",
        "Rare"
    ]

    # Imposer toutes les catégories permet d'obtenir les mêmes
    # colonnes même pour un seul passager.
    X["Embarked"] = pd.Categorical(
        X["Embarked"],
        categories=embarked_categories
    )

    X["Title"] = pd.Categorical(
        X["Title"],
        categories=title_categories
    )

    X = pd.get_dummies(
        X,
        columns=[
            "Embarked",
            "Title"
        ],
        dtype=int,
        drop_first=True
    )

    print("Colonnes après One-Hot :")
    print(X.columns.tolist())

    print("✓ One-Hot Encoding terminé")


    # ============================================================
    # 8. ALIGNEMENT AVEC LE MODÈLE
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 8 — ALIGNEMENT AVEC LE MODÈLE")
    print("=" * 70)

    # Sécurité : ajouter une colonne manquante avec 0
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Conserver uniquement les variables du modèle
    # et dans le même ordre.
    X_model = X[feature_columns].copy()

    print("Features finales :")
    print(X_model.columns.tolist())

    if X_model.columns.tolist() != feature_columns:
        raise ValueError(
            "ERREUR : les colonnes finales ne correspondent pas "
            "aux colonnes attendues par le modèle."
        )

    if X_model.shape[1] != len(feature_columns):
        raise ValueError(
            f"ERREUR : {X_model.shape[1]} features obtenues "
            f"au lieu de {len(feature_columns)}."
        )

    print("✓ Les 15 features sont présentes")
    print("✓ Les features sont dans le bon ordre")


    # ============================================================
    # 9. CONTRÔLES AVANT STANDARDISATION
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 9 — CONTRÔLES AVANT STANDARDISATION")
    print("=" * 70)

    print("Shape :", X_model.shape)

    if X_model.isna().any().any():
        missing_values = (
            X_model
            .isna()
            .sum()
        )

        missing_values = missing_values[
            missing_values > 0
        ]

        raise ValueError(
            "ERREUR : valeurs manquantes restantes avant scaler : "
            f"{missing_values.to_dict()}"
        )

    print("✓ Aucune valeur manquante")

    non_numeric_columns = [
        col
        for col in X_model.columns
        if not pd.api.types.is_numeric_dtype(
            X_model[col]
        )
    ]

    if non_numeric_columns:
        raise TypeError(
            "ERREUR : certaines features ne sont pas numériques : "
            f"{non_numeric_columns}"
        )

    print("✓ Toutes les features sont numériques")


    # ============================================================
    # 10. VÉRIFICATION DU SCALER
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 10 — VÉRIFICATION DU SCALER")
    print("=" * 70)

    print("Type du scaler :", type(scaler))

    if not hasattr(scaler, "mean_"):
        raise ValueError(
            "ERREUR : le scaler ne semble pas avoir été entraîné."
        )

    if not hasattr(scaler, "scale_"):
        raise ValueError(
            "ERREUR : le scaler ne possède pas scale_."
        )

    if not hasattr(scaler, "n_features_in_"):
        raise ValueError(
            "ERREUR : le scaler ne possède pas n_features_in_."
        )

    if scaler.n_features_in_ != len(feature_columns):
        raise ValueError(
            f"ERREUR : le scaler attend "
            f"{scaler.n_features_in_} features au lieu de "
            f"{len(feature_columns)}."
        )

    # Vérification de l'ordre si le scaler a mémorisé
    # les noms des colonnes
    if hasattr(scaler, "feature_names_in_"):

        scaler_columns = (
            scaler.feature_names_in_.tolist()
        )

        if scaler_columns != feature_columns:
            raise ValueError(
                "ERREUR : ordre des colonnes différent de celui "
                "utilisé lors de l'entraînement du scaler.\n"
                f"Scaler : {scaler_columns}\n"
                f"Pipeline : {feature_columns}"
            )

    print("✓ Scaler déjà entraîné")
    print("✓ Le scaler attend bien 15 features")


    # ============================================================
    # 11. STANDARDISATION
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 11 — STANDARDISATION")
    print("=" * 70)

    # IMPORTANT :
    # aucun fit ni fit_transform sur les nouvelles données
    X_scaled_array = scaler.transform(
        X_model
    )

    print("Type après scaler :", type(X_scaled_array))
    print("Shape après scaler :", X_scaled_array.shape)

    expected_shape = (
        len(X_model),
        len(feature_columns)
    )

    if X_scaled_array.shape != expected_shape:
        raise ValueError(
            f"ERREUR : shape attendue {expected_shape}, "
            f"shape obtenue {X_scaled_array.shape}."
        )

    print("✓ Shape compatible avec le modèle")


    # ============================================================
    # 12. DATAFRAME FINAL
    # ============================================================

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=feature_columns,
        index=X_model.index
    )

    print("\n" + "=" * 70)
    print("ÉTAPE 12 — DATASET FINAL")
    print("=" * 70)

    print(X_scaled)

    print("\nColonnes finales :")
    print(X_scaled.columns.tolist())

    print("\nShape finale :")
    print(X_scaled.shape)

    print("\nValeurs manquantes :")
    print(X_scaled.isna().sum().sum())

    if X_scaled.isna().any().any():
        raise ValueError(
            "ERREUR : X_scaled contient des valeurs manquantes."
        )

    if not np.isfinite(
        X_scaled.to_numpy()
    ).all():
        raise ValueError(
            "ERREUR : X_scaled contient des valeurs infinies "
            "ou non numériques."
        )

    print("\n✓ X_scaled est prêt pour le modèle.")

    return X_scaled





def validate_input(X):

    required_columns = [
        "Pclass",
        "Name", 
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Cabin",
        "Embarked"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    if X.empty:
        raise ValueError(
            "Les données sont vides."
        )