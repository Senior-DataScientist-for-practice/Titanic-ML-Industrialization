"""
Ce fichier décrit :
-   les données que l’API accepte ;
-   le type de chaque variable ;
-   la structure de la réponse.

Autrement dit :

Entrée attendue : 

Pclass
Name
Sex
Age
SibSp
Parch
Fare
Cabin
Embarked


Réponse renvoyée : 

prediction
label
survival_probability

""" 


from typing import Optional

from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    """
    Données brutes attendues pour un passager.

    Les colonnes existent toujours dans le schéma,
    mais leurs valeurs peuvent être nulles.
    Le pipeline ML se charge ensuite des imputations.
    """

    Pclass: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Classe du passager : 1, 2 ou 3."
    )

    Name: Optional[str] = Field(
        default=None,
        description="Nom du passager contenant éventuellement son titre."
    )

    Sex: Optional[str] = Field(
        default=None,
        description="Sexe du passager : male ou female."
    )

    Age: Optional[float] = Field(
        default=None,
        ge=0,
        description="Âge du passager."
    )

    SibSp: Optional[int] = Field(
        default=None,
        ge=0,
        description="Nombre de frères, sœurs ou conjoints à bord."
    )

    Parch: Optional[int] = Field(
        default=None,
        ge=0,
        description="Nombre de parents ou enfants à bord."
    )

    Fare: Optional[float] = Field(
        default=None,
        ge=0,
        description="Prix du billet."
    )

    Cabin: Optional[str] = Field(
        default=None,
        description="Numéro ou référence de cabine."
    )

    Embarked: Optional[str] = Field(
        default=None,
        description="Port d’embarquement : C, Q ou S."
    )


class PredictionResponse(BaseModel):
    """
    Structure de la réponse renvoyée par l’API.
    """

    prediction: int
    label: str
    survival_probability: float
