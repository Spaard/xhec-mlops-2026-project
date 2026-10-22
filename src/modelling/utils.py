"""Helpers to save and load Python objects (model, encoder) with pickle."""

import pickle
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, path: Path) -> None:
    """Pickle an object to the given path, creating the parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickled_object(path: Path) -> Any:
    """Load an object pickled with `pickle_object`.

    Only load files produced by this project: unpickling can execute arbitrary code.
    """
    with Path(path).open("rb") as file:
        return pickle.load(file)
