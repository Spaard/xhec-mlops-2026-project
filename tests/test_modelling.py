import pandas as pd
import pytest
from modelling.preprocessing import clean_data, load_data, split_data
from modelling.utils import load_pickled_object, pickle_object

DATA_PATH = "data/abalone.csv"


def make_abalone(**overrides: float | str) -> dict[str, float | str]:
    row = {
        "sex": "M",
        "length": 0.5,
        "diameter": 0.4,
        "height": 0.14,
        "whole_weight": 0.8,
        "shucked_weight": 0.35,
        "viscera_weight": 0.18,
        "shell_weight": 0.24,
        "rings": 10,
    }
    return row | overrides


def test_clean_data_drops_impossible_rows() -> None:
    df = pd.DataFrame(
        [
            make_abalone(),
            make_abalone(height=0),
            make_abalone(diameter=0.6),
            make_abalone(shell_weight=0.9),
        ]
    )
    assert len(clean_data(df)) == 1


def test_pickle_round_trip(tmp_path) -> None:
    path = tmp_path / "nested" / "object.pkl"
    pickle_object({"a": 1}, path)
    assert load_pickled_object(path) == {"a": 1}


@pytest.mark.skipif(
    not pd.io.common.file_exists(DATA_PATH), reason="dataset not downloaded"
)
def test_split_data_keeps_all_rows() -> None:
    df = clean_data(load_data(DATA_PATH))
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert "rings" not in X_train.columns
