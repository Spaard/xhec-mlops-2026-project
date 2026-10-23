"""Inference: turn validated abalones into predictions of the trained model."""

import pandas as pd
from sklearn.base import BaseEstimator

from modelling.predicting import predict
from web_service.lib.models import Abalone, Prediction

# The age of an abalone is its number of rings plus 1.5 years
YEARS_ADDED_TO_RINGS = 1.5


def run_inference(model: BaseEstimator, abalones: list[Abalone]) -> list[Prediction]:
    """Predict the number of rings and the age of each abalone."""
    features = pd.DataFrame([abalone.model_dump() for abalone in abalones])
    rings = predict(model, features)
    return [
        Prediction(rings=round(value, 2), age=round(value + YEARS_ADDED_TO_RINGS, 2))
        for value in rings.tolist()
    ]
