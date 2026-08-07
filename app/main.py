"""
Ce fichier :

-   crée l’application FastAPI ;
-   crée les routes /, /health et /predict ;
-   appelle ton pipeline ML existant.


"""

import pandas as pd

from fastapi import FastAPI, HTTPException

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
        # Conversion du modèle Pydantic en dictionnaire Python
        passenger_data = passenger.model_dump()

        # Conversion du dictionnaire en DataFrame d'une ligne
        passenger_df = pd.DataFrame(
            [passenger_data]
        )

        # Appel de ton pipeline existant
        prediction, probability, _ = predict(
            passenger_df,
            scaler,
            model
        )

        # Conversion des types NumPy en types Python classiques
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