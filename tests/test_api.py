import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

"""------------------------------------"""
def create_valid_passenger():

    return {
        "Pclass": 2,
        "Name": "Test, Mr. Passenger",
        "Sex": "male",
        "Age": 30,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 20.0,
        "Cabin": None,
        "Embarked": "S"
    }

"""------------------------------------"""
def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Titanic ML API"
    assert data["documentation"] == "/docs"

"""------------------------------------"""
def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "LogisticRegression"
    assert data["scaler"] == "StandardScaler"

"""------------------------------------"""
def test_predict_valid_passenger():

    passenger = create_valid_passenger()

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]

    assert data["label"] in [
        "Survivant",
        "Non survivant"
    ]

    assert (
        0
        <= data["survival_probability"]
        <= 1
    )

"""------------------------------------"""
def test_predict_with_missing_values():

    passenger = {
        "Pclass": 3,
        "Name": "Example, Mr. Unknown",
        "Sex": None,
        "Age": None,
        "SibSp": None,
        "Parch": None,
        "Fare": None,
        "Cabin": None,
        "Embarked": None
    }

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]

    assert (
        0
        <= data["survival_probability"]
        <= 1
    )

"""------------------------------------"""
def test_invalid_pclass():

    passenger = create_valid_passenger()

    passenger["Pclass"] = 7

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 422

"""------------------------------------"""
def test_negative_age():

    passenger = create_valid_passenger()

    passenger["Age"] = -10

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 422

"""------------------------------------"""
def test_invalid_age_type():

    passenger = create_valid_passenger()

    passenger["Age"] = "abc"

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 422

"""------------------------------------"""
def test_empty_request():

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 200