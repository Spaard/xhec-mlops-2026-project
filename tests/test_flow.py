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


def test_training_flow_saves_a_working_model(tmp_path, random_abalones) -> None:
    trainset_path = tmp_path / "abalone.csv"
    random_abalones.to_csv(trainset_path, index=False)
    model_path = tmp_path / "model.pkl"

    metrics = main(trainset_path, model_path)

    assert set(metrics) == {"rmse", "mae", "r2"}
    model = load_pickled_object(model_path)
    features = load_data(trainset_path).drop(columns="rings").head()
    assert len(model.predict(features)) == len(features)
