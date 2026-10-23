import pytest
from fastapi.testclient import TestClient

from modelling.preprocessing import clean_data, load_data
from modelling.training import train_model
from web_service.main import app

ABALONE = {
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory, random_abalones):
    """API client with a model trained on random abalones."""
    trainset_path = tmp_path_factory.mktemp("data") / "abalone.csv"
    random_abalones.to_csv(trainset_path, index=False)
    df = clean_data(load_data(trainset_path))
    app.state.model = train_model(df.drop(columns="rings"), df["rings"])
    yield TestClient(app)
    app.state.model = None


def test_home(client) -> None:
    response = client.get("/")
    assert response.status_code == 200


def test_predict_returns_one_prediction_per_abalone(client) -> None:
    response = client.post("/predict", json={"abalones": [ABALONE, ABALONE]})
    assert response.status_code == 201
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["age"] == pytest.approx(predictions[0]["rings"] + 1.5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sex": "X"},
        {"height": 0},
        {"diameter": 0.5},
        {"shell_weight": 0.6},
        {"height": 0.4},
        {"unknown_feature": 1},
    ],
)
def test_predict_rejects_invalid_abalones(client, overrides) -> None:
    response = client.post("/predict", json={"abalones": [ABALONE | overrides]})
    assert response.status_code == 422


def test_predict_rejects_empty_requests(client) -> None:
    response = client.post("/predict", json={"abalones": []})
    assert response.status_code == 422


def test_predict_without_model_returns_503() -> None:
    app.state.model = None
    response = TestClient(app).post("/predict", json={"abalones": [ABALONE]})
    assert response.status_code == 503
