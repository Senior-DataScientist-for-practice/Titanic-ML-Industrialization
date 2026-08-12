"""
API FastAPI pour le modèle Titanic.

Ce fichier :
- crée l'application FastAPI ;
- expose les routes classiques /, /health et /predict ;
- expose les routes SageMaker /ping et /invocations ;
- utilise le pipeline ML existant ;
- écrit des logs explicites visibles dans CloudWatch.
"""

import logging

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PassengerInput, PredictionResponse
from src.predict import model, predict, scaler


# ============================================================
# LOGGING
# ============================================================

# Les logs écrits sur stdout/stderr par le conteneur SageMaker
# sont récupérés par CloudWatch.
logger = logging.getLogger("titanic-api")
logger.setLevel(logging.INFO)

# Évite d'ajouter plusieurs handlers si le module est rechargé.
if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="Titanic ML API",
    description=(
        "API de prédiction de survie des passagers du Titanic "
        "utilisant un pipeline Machine Learning industrialisé."
    ),
    version="1.0.0",
)


# ============================================================
# FONCTION COMMUNE DE PRÉDICTION
# ============================================================

def run_prediction(passenger_data: dict) -> dict:
    """
    Exécute le pipeline de prédiction Titanic.

    Cette fonction est utilisée à la fois par :
    - /predict
    - /invocations
    """

    logger.info("Début de la prédiction.")

    # Ne pas logger le nom du passager :
    # on garde uniquement les informations utiles au monitoring.
    logger.info(
        "Input reçu : Pclass=%s Sex=%s Age=%s SibSp=%s "
        "Parch=%s Fare=%s Embarked=%s",
        passenger_data.get("Pclass"),
        passenger_data.get("Sex"),
        passenger_data.get("Age"),
        passenger_data.get("SibSp"),
        passenger_data.get("Parch"),
        passenger_data.get("Fare"),
        passenger_data.get("Embarked"),
    )

    passenger_df = pd.DataFrame([passenger_data])

    prediction, probability, _ = predict(
        passenger_df,
        scaler,
        model,
    )

    predicted_class = int(prediction[0])
    survival_probability = float(probability[0, 1])

    label = (
        "Survivant"
        if predicted_class == 1
        else "Non survivant"
    )

    logger.info(
        "Prédiction terminée : prediction=%s "
        "label=%s survival_probability=%.6f",
        predicted_class,
        label,
        survival_probability,
    )

    return {
        "prediction": predicted_class,
        "label": label,
        "survival_probability": survival_probability,
    }


# ============================================================
# ROUTE D'ACCUEIL
# ============================================================

@app.get("/")
def root():
    """
    Route d'accueil.
    """

    return {
        "message": "Titanic ML API",
        "documentation": "/docs",
    }


# ============================================================
# HEALTH CHECK FASTAPI
# ============================================================

@app.get("/health")
def health():
    """
    Vérifie que l'API, le modèle et le scaler sont chargés.
    """

    return {
        "status": "ok",
        "model": type(model).__name__,
        "scaler": type(scaler).__name__,
    }


# ============================================================
# ROUTE DE PRÉDICTION FASTAPI
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_survival(passenger: PassengerInput):
    """
    Route de prédiction classique FastAPI.
    """

    logger.info("POST /predict reçu.")

    try:
        passenger_data = passenger.model_dump()

        result = run_prediction(passenger_data)

        return PredictionResponse(**result)

    except (ValueError, TypeError, KeyError) as error:

        logger.warning(
            "Erreur de validation sur /predict : %s",
            str(error),
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception:

        logger.exception(
            "Erreur interne pendant /predict."
        )

        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction.",
        )


# ============================================================
# SAGEMAKER HEALTH CHECK
# ============================================================

@app.get("/ping")
def ping():
    """
    Health check utilisé automatiquement par SageMaker.
    """

    return {
        "status": "ok"
    }


# ============================================================
# SAGEMAKER INFERENCE
# ============================================================

@app.post("/invocations")
def invocations(passenger: PassengerInput):
    """
    Endpoint d'inférence utilisé par SageMaker Runtime.
    """

    logger.info(
        "========== SAGEMAKER INVOCATION RECEIVED =========="
    )

    try:
        passenger_data = passenger.model_dump()

        result = run_prediction(passenger_data)

        logger.info(
            "SageMaker invocation SUCCESS | "
            "prediction=%s | probability=%.6f",
            result["prediction"],
            result["survival_probability"],
        )

        logger.info(
            "========== SAGEMAKER INVOCATION COMPLETED =========="
        )

        return result

    except (ValueError, TypeError, KeyError) as error:

        logger.warning(
            "SageMaker invocation INVALID INPUT | %s",
            str(error),
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception:

        logger.exception(
            "SageMaker invocation FAILED."
        )

        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction.",
        )