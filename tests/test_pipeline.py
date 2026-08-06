import sys
from pathlib import Path
import pytest

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import pandas as pd

from src.preprocessing import validate_input
from src.predict import predict
from src.predict import scaler
from src.predict import model



def create_valid_passenger():

    return pd.DataFrame({
        "Age": [30],
        "Sex": ["male"],
        "Name": ["Test, Mr. Passenger"],
        "SibSp": [0],
        "Fare": [20.0],
        "Parch": [0],
        "Cabin": [None],
        "Embarked": ["S"],
        "Pclass": [2]
    })


def test_valid_input():

    passenger = create_valid_passenger()

    validate_input(passenger)



def test_missing_column():

    passenger = create_valid_passenger()

    passenger = passenger.drop(
        columns=["Embarked"]
    )

    with pytest.raises(ValueError):
        validate_input(passenger)



def test_prediction():

    passenger = create_valid_passenger()

    prediction, probability, X_scaled = predict(
        passenger,
        scaler,
        model
    )

    assert prediction[0] in [0, 1]



def test_prediction_probability():

    passenger = create_valid_passenger()

    prediction, probability, X_scaled = predict(
        passenger,
        scaler,
        model
    )
    print(probability)
    assert 0 <= probability[0][1] <= 1


def test_missing_values():

    passenger = create_valid_passenger()

    passenger["Age"] = None
    passenger["Embarked"] = None

    prediction, probability, X_scaled = predict(
        passenger,
        scaler,
        model
    )
    print("passager avec infos None: ", "prediction : ", prediction, "probabilité : ", probability)
    assert prediction[0] in [0, 1]


def test_empty_dataframe():
    passenger = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_input(passenger)

def test_batch_prediction():
    passenger_1 = create_valid_passenger()
    passenger_2 = create_valid_passenger()

    passengers = pd.concat(
        [passenger_1, passenger_2],
        ignore_index=True
    )

    prediction, probability, X_scaled = predict(
        passengers,
        scaler,
        model
    )

    assert len(prediction) == 2
    assert probability.shape == (2, 2)
    assert X_scaled.shape == (2, 15)


def test_scaled_output_shape():
    passenger = create_valid_passenger()

    prediction, probability, X_scaled = predict(
        passenger,
        scaler,
        model
    )

    assert X_scaled.shape == (1, 15)

def test_no_missing_values_after_preprocessing():
    passenger = create_valid_passenger()

    passenger["Age"] = None
    passenger["Fare"] = None
    passenger["Embarked"] = None
    passenger["Sex"] = None

    prediction, probability, X_scaled = predict(
        passenger,
        scaler,
        model
    )

    assert X_scaled.isna().sum().sum() == 0