"""Helpers of the web service."""

import pickle
from pathlib import Path
from typing import Any


def load_object(path: Path) -> Any:
    """Load a pickled object, such as the trained model.

    Only load files produced by this project: unpickling can execute arbitrary code.
    """
    with Path(path).open("rb") as file:
        return pickle.load(file)
