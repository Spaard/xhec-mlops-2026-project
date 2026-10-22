import numpy as np
import pandas as pd
import pytest


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


@pytest.fixture(scope="session")
def random_abalones() -> pd.DataFrame:
    """200 random abalones in the format of the Kaggle file."""
    return make_dataset(200)
