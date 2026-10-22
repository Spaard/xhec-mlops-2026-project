"""Data loading, cleaning, splitting and feature engineering of the abalone data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "rings"
CATEGORICAL_FEATURE = "sex"
PART_WEIGHTS = ["shucked_weight", "viscera_weight", "shell_weight"]
MEASUREMENTS = ["length", "diameter", "height", "whole_weight", *PART_WEIGHTS]

# Physical consistency rules (see the EDA): height / length must stay in this range
MIN_HEIGHT_RATIO = 0.05
MAX_HEIGHT_RATIO = 0.6

TEST_SIZE = 0.2
N_STRATIFICATION_BINS = 10
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load the abalone dataset and convert its column names to snake_case."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def find_invalid_rows(df: pd.DataFrame) -> pd.Series:
    """Flag the rows that violate a physical constraint of the measurements."""
    dimensions = df[["length", "diameter", "height"]]
    height_ratio = df["height"] / df["length"]
    return (
        (dimensions <= 0).any(axis=1)
        | (df["diameter"] > df["length"])
        | df[PART_WEIGHTS].gt(df["whole_weight"], axis=0).any(axis=1)
        | ~height_ratio.between(MIN_HEIGHT_RATIO, MAX_HEIGHT_RATIO)
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the physically impossible rows."""
    return df[~find_invalid_rows(df)].reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test sets, stratified on the target."""
    features = df.drop(columns=TARGET)
    target = df[TARGET]
    target_bins = pd.qcut(
        target, N_STRATIFICATION_BINS, labels=False, duplicates="drop"
    )
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target_bins,
        random_state=RANDOM_STATE,
    )


def add_weight_ratios(features: pd.DataFrame) -> pd.DataFrame:
    """Add the share of each part (meat, viscera, shell) in the whole weight."""
    # NumPy arrays are assigned by position, which stays correct even if the index
    # has duplicates (a join on the index would multiply the rows)
    whole_weight = features["whole_weight"].to_numpy()
    return features.assign(
        **{
            f"{part}_ratio": features[part].to_numpy() / whole_weight
            for part in PART_WEIGHTS
        }
    )


def log_measurements(features: pd.DataFrame) -> pd.DataFrame:
    """Replace the measurements by their logarithm, which linearises them."""
    return features.assign(
        **{column: np.log(features[column].to_numpy()) for column in MEASUREMENTS}
    )


def build_preprocessor(log_features: bool) -> Pipeline:
    """Build the feature engineering, one-hot encoding and scaling steps."""
    steps = [FunctionTransformer(add_weight_ratios)]
    if log_features:
        steps.append(FunctionTransformer(log_measurements))
    encoder = ColumnTransformer(
        [
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [CATEGORICAL_FEATURE],
            )
        ],
        remainder=StandardScaler(),
        verbose_feature_names_out=False,
    )
    steps.append(encoder)
    pipeline = make_pipeline(*steps)
    pipeline.set_output(transform="pandas")
    return pipeline
