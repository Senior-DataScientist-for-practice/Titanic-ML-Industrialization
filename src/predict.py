import joblib
from pathlib import Path
from .preprocessing import prepare_new_data_for_prediction, validate_input

BASE_DIR = Path(__file__).resolve().parent.parent

scaler = joblib.load(
    BASE_DIR / "models" / "standard_scaler.pkl"
)

model = joblib.load(
    BASE_DIR / "models" / "logistic_regression_final.pkl"
)

preprocessing_info = joblib.load(
    BASE_DIR / "models" / "preprocessing_info.pkl"
)

training_medians = preprocessing_info["training_medians"]

training_modes = preprocessing_info["training_modes"]



def predict(X, scaler, model):

    # Validation
    validate_input(X)

    # Préparation
    X_scaled = prepare_new_data_for_prediction(
        X,scaler)

    # Prédiction
    prediction = model.predict(X_scaled)

    # Probabilité
    probability = model.predict_proba(X_scaled)

    return prediction, probability, X_scaled