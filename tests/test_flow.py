import numpy as np
import pandas as pd
import pytest
from prefect.testing.utilities import prefect_test_harness

from modelling.main import main
from modelling.preprocessing import load_data
from modelling.utils import load_pickled_object


@pytest.fixture(scope="module", autouse=True)
def prefect_test_server():
    """Run the flows of this module against a temporary Prefect database."""
    with prefect_test_harness():
        yield


def make_dataset(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Random abalones, in the format of the Kaggle file, that pass the cleaning."""
    rng = np.random.default_rng(seed)
    length = rng.uniform(0.2, 0.7, n_rows)
    whole_weight = rng.uniform(0.1, 2.0, n_rows)
    return pd.DataFrame(
        {
            "Sex": rng.choice(["M", "F", "I"], n_rows),
            "Length": length,
            "Diameter": length * rng.uniform(0.7, 0.85, n_rows),
            "Height": length * rng.uniform(0.2, 0.35, n_rows),
            "Whole weight": whole_weight,
            "Shucked weight": whole_weight * rng.uniform(0.3, 0.5, n_rows),
            "Viscera weight": whole_weight * rng.uniform(0.1, 0.25, n_rows),
            "Shell weight": whole_weight * rng.uniform(0.2, 0.35, n_rows),
            "Rings": rng.integers(3, 20, n_rows),
        }
    )


def test_training_flow_saves_a_working_model(tmp_path) -> None:
    trainset_path = tmp_path / "abalone.csv"
    make_dataset(200).to_csv(trainset_path, index=False)
    model_path = tmp_path / "model.pkl"

    metrics = main(trainset_path, model_path)

    assert set(metrics) == {"rmse", "mae", "r2"}
    model = load_pickled_object(model_path)
    features = load_data(trainset_path).drop(columns="rings").head()
    assert len(model.predict(features)) == len(features)
