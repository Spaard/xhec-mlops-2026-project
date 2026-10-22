"""Inference: load the pickled model and predict the number of rings."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from modelling.utils import load_pickled_object

MODEL_PATH = Path(__file__).parents[1] / "web_service" / "local_objects" / "model.pkl"


def load_model(path: Path = MODEL_PATH) -> BaseEstimator:
    """Load the trained model (preprocessing included) from a pickle file."""
    return load_pickled_object(path)


def predict(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Predict the number of rings of abalones described by snake_case columns."""
    return model.predict(features)
