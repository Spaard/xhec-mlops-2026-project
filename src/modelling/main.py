"""Training flow: read the data, clean it, train the model and save it."""

import argparse
from pathlib import Path

from modelling.predicting import MODEL_PATH
from modelling.preprocessing import clean_data, load_data, split_data
from modelling.training import evaluate_model, train_model
from modelling.utils import pickle_object


def main(trainset_path: Path, model_path: Path = MODEL_PATH) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    The pickled object is the whole pipeline (feature engineering, encoder and
    stacking ensemble), so the prediction API needs no separate encoder.
    """
    df = clean_data(load_data(trainset_path))
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print("Test metrics:", ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    pickle_object(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
