# 🚢 Industrialisation d'un Pipeline de Machine Learning – Titanic Survival Prediction

## 📖 Présentation

Ce projet a pour objectif de construire puis d'industrialiser un pipeline complet de Machine Learning permettant de prédire la survie des passagers du Titanic à partir de leurs caractéristiques individuelles.

Contrairement à une simple étude exploratoire, ce projet couvre l'ensemble du cycle de vie d'un modèle de Machine Learning :

* exploration et compréhension des données ;
* préparation et nettoyage des données ;
* feature engineering ;
* entraînement et comparaison de plusieurs modèles ;
* optimisation des hyperparamètres ;
* interprétation des résultats ;
* industrialisation du pipeline de prédiction ;
* mise en place de tests unitaires garantissant la robustesse du système.

Le projet est entièrement développé en Python en utilisant les bonnes pratiques d'industrialisation.

---

# 🎯 Objectifs

Les principaux objectifs sont :

* construire un modèle de classification performant ;
* reproduire exactement le pipeline de prétraitement lors de la prédiction ;
* gérer automatiquement les données manquantes ;
* garantir la cohérence des variables utilisées par le modèle ;
* sauvegarder les artefacts du pipeline ;
* automatiser les contrôles grâce à des tests unitaires.

---

# 📂 Structure du projet


Titanic-ML/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression_final.pkl
│   ├── standard_scaler.pkl
│   └── preprocessing_info.pkl
│
├── notebooks/
│   ├── 01_Exploration.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modelisation.ipynb
│   ├── 04_Optimisation.ipynb
│   ├── 05_Interpretabilite.ipynb
│   ├── 06_Evaluation.ipynb
│   └── 07_Industrialisation.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── predict.py
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
│
└── README.md


---

# 📊 Description des notebooks

## Notebook 01 – Exploration des données

Analyse exploratoire du jeu de données :

* statistiques descriptives ;
* valeurs manquantes ;
* distributions ;
* corrélations ;
* compréhension métier des variables.

---

## Notebook 02 – Prétraitement des données

Construction du pipeline de préparation :

* nettoyage ;
* gestion des valeurs manquantes ;
* Feature Engineering ;
* création des variables dérivées ;
* encodage des variables catégorielles ;
* standardisation.

---

## Notebook 03 – Modélisation

Construction des premiers modèles :

* Régression Logistique
* Random Forest
* KNN
* XGBoost

Comparaison des performances.

---

## Notebook 04 – Optimisation

Recherche des meilleurs hyperparamètres à l'aide de GridSearchCV.

Évaluation des performances obtenues.

---

## Notebook 05 – Interprétabilité

Analyse du comportement du modèle grâce aux méthodes d'interprétation (importance des variables, SHAP, etc.).

---

## Notebook 06 – Évaluation finale

Validation des performances du modèle retenu :

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Matrice de confusion

---

## Notebook 07 – Industrialisation

Développement du pipeline complet de prédiction :

* validation des données d'entrée ;
* reproduction exacte du prétraitement utilisé pendant l'entraînement ;
* gestion automatique des valeurs manquantes ;
* standardisation ;
* prédiction ;
* calcul des probabilités ;
* automatisation des tests.

---

# 🧠 Pipeline de Machine Learning

Le pipeline complet suit les étapes suivantes :


-   Nouvelles données
-   Validation des entrées
-   Gestion des valeurs manquantes
-   Création des variables dérivées
-   Encodage des variables
-   Alignement des features
-   Standardisation
-   Prédiction
-   Probabilité de survie


---

# ⚙️ Technologies utilisées

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* PyTest
* Jupyter Notebook

---

# 💾 Artefacts sauvegardés

À l'issue de l'entraînement, les éléments suivants sont sauvegardés :

## Modèle


models/logistic_regression_final.pkl

Contient le modèle final entraîné.

---

## StandardScaler


models/standard_scaler.pkl

Utilisé pour standardiser les nouvelles observations.

---

## Informations de prétraitement


models/preprocessing_info.pkl

Contient les statistiques calculées sur le jeu d'entraînement nécessaires au pipeline :

* médianes des variables numériques ;
* modalités les plus fréquentes des variables catégorielles.

Ces informations garantissent que les nouvelles données sont préparées exactement comme lors de l'entraînement.

---

# 🚀 Utilisation

## Installation

Créer un environnement virtuel :

```bash
python -m venv venv
```

Activation :

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Installer les dépendances :
pip install -r requirements.txt


## ▶️ Exemple de prédiction

from src.predict import predict, scaler, model

prediction, probability, X_scaled = predict(
    new_passenger,
    scaler,
    model
)

Le pipeline :
-   valide les données ;
-   applique automatiquement le prétraitement ;
-   standardise les variables ;
-   effectue la prédiction ;
-   retourne également la probabilité associée.


## 🧪 Tests unitaires

Les tests sont réalisés avec PyTest.

Lancement : pytest -v

Les principaux scénarios couverts sont :

-   validation des données d'entrée ;
-   détection des colonnes manquantes ;
-   prédiction d'un passager ;
-   calcul des probabilités ;
-   gestion des valeurs manquantes ;
-   DataFrame vide ;
-   prédiction sur plusieurs passagers ;
-   vérification du format des données standardisées ;
-   absence de valeurs manquantes après le prétraitement.

## 📈 Performances
Le modèle retenu est une Régression Logistique optimisée.

Les performances obtenues sont évaluées à l'aide de :

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Matrice de confusion

Les résultats détaillés sont présentés dans les notebooks d'évaluation.

## ✅ Bonnes pratiques mises en œuvre

-   séparation entre notebooks et code métier ;
-   sauvegarde des artefacts du pipeline ;
-   réutilisation du même prétraitement en entraînement et en prédiction ;
-   validation systématique des entrées ;
-   gestion robuste des données manquantes ;
-   automatisation des tests unitaires ;
-   architecture modulaire.