"""Definition, training and evaluation of the final model (a stacking ensemble)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.svm import SVR

from modelling.preprocessing import RANDOM_STATE, build_preprocessor

STACKING_CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
RIDGE_ALPHAS = np.logspace(-3, 3, 20)


def make_model(
    regressor: BaseEstimator, log_features: bool = False, log_target: bool = False
) -> BaseEstimator:
    """Chain the preprocessing and a regressor, optionally predicting log(rings)."""
    model = make_pipeline(build_preprocessor(log_features), regressor)
    if log_target:
        # Fit on log(rings), which is almost symmetric, and predict exp of the output
        return TransformedTargetRegressor(model, func=np.log, inverse_func=np.exp)
    return model


def build_base_models() -> dict[str, BaseEstimator]:
    """Build the four base models with the hyperparameters tuned in the notebook."""
    return {
        "svr (log, tuned)": make_model(
            SVR(C=7.819, epsilon=1.156, gamma=0.0684), log_features=True
        ),
        "ridge on splines (log, tuned)": make_model(
            make_pipeline(
                SplineTransformer(n_knots=4, degree=3), RidgeCV(alphas=RIDGE_ALPHAS)
            ),
            log_features=True,
        ),
        "random forest (tuned)": make_model(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=15,
                max_features=0.596,
                random_state=RANDOM_STATE,
            )
        ),
        "gradient boosting (tuned)": make_model(
            HistGradientBoostingRegressor(
                max_iter=1000,
                early_stopping=True,
                learning_rate=0.0335,
                max_leaf_nodes=15,
                min_samples_leaf=48,
                l2_regularization=0.171,
                max_features=0.345,
                random_state=RANDOM_STATE,
            )
        ),
    }


def build_model() -> StackingRegressor:
    """Combine the base models with a Ridge regression trained on their predictions."""
    return StackingRegressor(
        estimators=list(build_base_models().items()),
        final_estimator=RidgeCV(alphas=RIDGE_ALPHAS),
        cv=STACKING_CV,
        n_jobs=-1,
    )


def train_model(features: pd.DataFrame, target: pd.Series) -> StackingRegressor:
    """Fit the final model, preprocessing included, on the given training data."""
    return build_model().fit(features, target)


def evaluate_model(
    model: BaseEstimator, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Compute the regression metrics of a model on the given data."""
    predictions = model.predict(features)
    return {
        "rmse": root_mean_squared_error(target, predictions),
        "mae": mean_absolute_error(target, predictions),
        "r2": r2_score(target, predictions),
    }
