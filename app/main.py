"""
Ce fichier :

- crée l’application FastAPI ;
- crée les routes locales /, /health et /predict ;
- ajoute les routes SageMaker /ping et /invocations ;
- appelle le pipeline ML existant.
"""

import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.schemas import PassengerInput, PredictionResponse
from src.predict import model, predict, scaler


app = FastAPI(
    title="Titanic ML API",
    description=(
        "API de prédiction de survie des passagers du Titanic "
        "utilisant un pipeline Machine Learning industrialisé."
    ),
    version="1.0.0"
)


@app.get("/")
def root():
    """
    Route d'accueil.
    """
    return {
        "message": "Titanic ML API",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    """
    Vérifie que l'API, le modèle et le scaler sont chargés.
    """
    return {
        "status": "ok",
        "model": type(model).__name__,
        "scaler": type(scaler).__name__
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_survival(passenger: PassengerInput):
    """
    Prédit la survie d'un passager.
    """

    try:
        passenger_data = passenger.model_dump()

        passenger_df = pd.DataFrame(
            [passenger_data]
        )

        prediction, probability, _ = predict(
            passenger_df,
            scaler,
            model
        )

        predicted_class = int(
            prediction[0]
        )

        survival_probability = float(
            probability[0, 1]
        )

        label = (
            "Survivant"
            if predicted_class == 1
            else "Non survivant"
        )

        return PredictionResponse(
            prediction=predicted_class,
            label=label,
            survival_probability=survival_probability
        )

    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction."
        ) from error


# ============================================================
# Routes spécifiques à Amazon SageMaker
# ============================================================

@app.get("/ping")
def sagemaker_ping():
    """
    Health check utilisé par SageMaker.

    SageMaker considère le conteneur comme sain
    lorsque cette route retourne HTTP 200.
    """

    if model is None or scaler is None:
        return Response(
            status_code=500
        )

    return Response(
        status_code=200
    )


@app.post("/invocations")
async def sagemaker_invocations(request: Request):
    """
    Endpoint d'inférence utilisé par SageMaker.

    Le body attendu est un JSON représentant un passager.
    """

    try:
        payload = await request.json()

        # Validation avec le même schéma Pydantic que /predict
        passenger = PassengerInput(**payload)

        passenger_data = passenger.model_dump()

        passenger_df = pd.DataFrame(
            [passenger_data]
        )

        prediction, probability, _ = predict(
            passenger_df,
            scaler,
            model
        )

        predicted_class = int(
            prediction[0]
        )

        survival_probability = float(
            probability[0, 1]
        )

        label = (
            "Survivant"
            if predicted_class == 1
            else "Non survivant"
        )

        return JSONResponse(
            content={
                "prediction": predicted_class,
                "label": label,
                "survival_probability": survival_probability
            },
            status_code=200
        )

    except (ValueError, TypeError, KeyError) as error:
        return JSONResponse(
            content={
                "error": str(error)
            },
            status_code=400
        )

    except Exception:
        return JSONResponse(
            content={
                "error": "Erreur interne lors de la prédiction."
            },
            status_code=500
        )