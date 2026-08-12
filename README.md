# Titanic ML Industrialization

Projet de Machine Learning visant à industrialiser un modèle de prédiction de survie sur le dataset Titanic, depuis l'entraînement local jusqu'au déploiement cloud sur AWS SageMaker avec CI/CD GitHub Actions.

## Objectif

L'objectif de ce projet est de transformer un modèle de Machine Learning classique en une application industrialisée, testable, conteneurisée et déployable automatiquement sur AWS.

Le projet couvre notamment :

* preprocessing et feature engineering ;
* entraînement et sérialisation du modèle ;
* tests automatisés avec `pytest` ;
* API REST avec FastAPI ;
* conteneurisation avec Docker ;
* stockage d'artefacts dans Amazon S3 ;
* stockage des images Docker dans Amazon ECR ;
* déploiement sur Amazon SageMaker ;
* monitoring avec Amazon CloudWatch ;
* authentification sécurisée GitHub → AWS avec OIDC ;
* pipeline CI/CD avec GitHub Actions.

---

## Architecture

```text
                        GitHub
                           │
                        git push
                           │
                           ▼
                    GitHub Actions
                 ┌─────────┴─────────┐
                 │                   │
              pytest            Docker Build
                                     │
                                     ▼
                              AWS OIDC / IAM
                                     │
                                     ▼
                                    ECR
                                     │
                                     ▼
                               SageMaker Model
                                     │
                                     ▼
                              Endpoint Config
                                     │
                                     ▼
                                  Endpoint
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
                Prediction                       CloudWatch

Amazon S3
   │
   ├── Dataset
   └── Model artifacts
```

---

## Structure du projet

```text
Titanic_Classification/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── entrypoint.sh
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression_final.pkl
│   ├── standard_scaler.pkl
│   ├── preprocessing_info.pkl
│   └── features.pkl
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   └── predict.py
│
├── tests/
│   ├── test_api.py
│   └── ...
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline Machine Learning

Le pipeline réalise notamment :

1. validation des données d'entrée ;
2. traitement des valeurs manquantes ;
3. création de variables dérivées ;
4. encodage des variables catégorielles ;
5. mise à l'échelle des variables ;
6. prédiction avec le modèle entraîné ;
7. calcul de la probabilité de survie.

Le modèle final utilisé dans l'API est une régression logistique sérialisée dans :

```text
models/logistic_regression_final.pkl
```

Le scaler et les informations nécessaires au preprocessing sont également sauvegardés sous forme d'artefacts.

---

## API FastAPI

L'API expose plusieurs routes.

### API classique

```text
GET  /
GET  /health
POST /predict
```

### Compatibilité SageMaker

```text
GET  /ping
POST /invocations
```

`/ping` est utilisé par SageMaker pour vérifier la santé du conteneur.

`/invocations` est utilisé par SageMaker Runtime pour envoyer les requêtes de prédiction.

Exemple de requête :

```json
{
  "Pclass": 3,
  "Name": "Braund, Mr. Owen Harris",
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Exemple de réponse :

```json
{
  "prediction": 0,
  "label": "Non survivant",
  "survival_probability": 0.077689
}
```

---

## Tests automatisés

Les tests sont exécutés avec :

```bash
pytest -v
```

Ils couvrent notamment :

* les routes API ;
* les prédictions valides ;
* les valeurs manquantes ;
* les types invalides ;
* la validation des variables ;
* le comportement du pipeline.

Les tests sont exécutés automatiquement dans GitHub Actions avant toute étape Docker ou AWS.

---

## Docker

L'application est conteneurisée avec Docker.

Le conteneur SageMaker écoute sur :

```text
8080
```

Le conteneur prend également en charge le lancement SageMaker avec :

```text
serve
```

grâce à :

```text
app/entrypoint.sh
```

Construction locale :

```bash
docker build -t titanic-ml-api .
```

Test local :

```bash
docker run --rm -p 8080:8080 titanic-ml-api serve
```

Health check :

```bash
curl -i http://127.0.0.1:8080/ping
```

---

## AWS

### Amazon S3

S3 est utilisé pour stocker :

```text
data/raw/
models/
```

Le bucket est configuré avec :

* blocage de l'accès public ;
* Object Ownership `BucketOwnerEnforced` ;
* versioning activé.

### Amazon ECR

ECR contient l'image Docker utilisée par SageMaker.

Repository :

```text
titanic-dev-api
```

Les images sont construites pour :

```text
linux/amd64
```

avec un manifeste Docker compatible SageMaker.

### Amazon SageMaker

Le projet utilise SageMaker pour :

* créer une définition de modèle ;
* créer une configuration d'endpoint ;
* déployer un endpoint d'inférence ;
* effectuer des prédictions avec SageMaker Runtime.

Le conteneur utilise :

```text
GET /ping
POST /invocations
```

### Amazon CloudWatch

CloudWatch permet d'observer :

* les health checks SageMaker ;
* les appels d'inférence ;
* les prédictions ;
* les erreurs éventuelles du conteneur.

Exemples de logs :

```text
SAGEMAKER INVOCATION RECEIVED
Début de la prédiction
Prédiction terminée
SageMaker invocation SUCCESS
SAGEMAKER INVOCATION COMPLETED
```

---

## Sécurité AWS

Le projet sépare plusieurs identités AWS.

```text
Root AWS
    │
    └── administration exceptionnelle

ml-engineer-dev
    │
    └── accès AWS CLI pour le développement

GitHubActionsTitanicRole
    │
    └── rôle assumé par GitHub Actions via OIDC

TitanicSageMakerExecutionRole
    │
    └── rôle utilisé par SageMaker
```

Le compte root est protégé par MFA.

GitHub Actions n'utilise pas de clés AWS permanentes.

L'authentification GitHub → AWS repose sur OpenID Connect :

```text
GitHub Actions
      ↓
OIDC Token
      ↓
AWS STS
      ↓
GitHubActionsTitanicRole
```

La trust policy est limitée au repository GitHub et à la branche `main`.

---

## CI/CD

Le workflow GitHub Actions est défini dans :

```text
.github/workflows/ci.yml
```

Le pipeline exécute :

```text
git push main
      ↓
pytest
      ↓
Docker build
      ↓
Authentification AWS via OIDC
      ↓
Push de l'image vers ECR
      ↓
Création du modèle SageMaker
      ↓
Création de l'EndpointConfig
      ↓
Création ou mise à jour de l'endpoint
      ↓
Attente du statut InService
```

Une Pull Request exécute les tests sans déclencher le déploiement AWS.

---

## Exécution locale

Créer un environnement Python :

```bash
python -m venv venv
```

Activer l'environnement :

```bash
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'API :

```bash
uvicorn app.main:app --reload
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

---

## Lancer les tests

```bash
pytest -v
```

---

## Technologies utilisées

### Data Science

* Python
* Pandas
* NumPy
* scikit-learn

### API

* FastAPI
* Pydantic
* Uvicorn

### Tests

* pytest
* FastAPI TestClient

### DevOps / MLOps

* Git
* GitHub
* GitHub Actions
* Docker

### AWS

* IAM
* STS
* OIDC
* S3
* ECR
* SageMaker
* CloudWatch

---

## Compétences mises en œuvre

Ce projet met en pratique :

* industrialisation d'un pipeline ML ;
* structuration d'un projet Python ;
* tests automatisés ;
* développement d'une API d'inférence ;
* conteneurisation ;
* déploiement cloud ;
* gestion IAM ;
* authentification OIDC ;
* CI/CD ;
* monitoring d'un modèle en production ;
* gestion des artefacts ML ;
* reproductibilité du déploiement.

---

## Statut du projet

Le pipeline complet a été validé de bout en bout :

```text
Code
✅
Tests
✅
FastAPI
✅
Docker
✅
GitHub Actions
✅
AWS OIDC
✅
S3
✅
ECR
✅
SageMaker
✅
Inference Endpoint
✅
CloudWatch
✅
CI/CD
✅
```

L'endpoint SageMaker peut être supprimé après les tests afin d'éviter de conserver des ressources de calcul inutiles.
