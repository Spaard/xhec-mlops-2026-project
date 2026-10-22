"""Training flow: read the data, clean it, train the model and save it.

The flow and each of its steps are tracked by Prefect, so every training run can be
followed in the Prefect UI. The steps are the functions of the other modules, turned
into Prefect tasks here so that those modules stay usable without Prefect (in the
tests and in the prediction API).
"""

import argparse
from pathlib import Path

from prefect import flow, get_run_logger, task

from modelling.predicting import MODEL_PATH
from modelling.preprocessing import clean_data, load_data, split_data
from modelling.training import evaluate_model, train_model
from modelling.utils import pickle_object

DATA_PATH = Path(__file__).parents[2] / "data" / "abalone.csv"

load_data_task = task(load_data)
clean_data_task = task(clean_data)
split_data_task = task(split_data)
train_model_task = task(train_model)
evaluate_model_task = task(evaluate_model)
pickle_object_task = task(pickle_object)


@flow(name="train-model")
def main(
    trainset_path: Path = DATA_PATH, model_path: Path = MODEL_PATH
) -> dict[str, float]:
    """Train a model using the data at the given path and save the model (pickle).

    The pickled object is the whole pipeline (feature engineering, encoder and
    stacking ensemble), so the prediction API needs no separate encoder.
    """
    logger = get_run_logger()
    df = clean_data_task(load_data_task(trainset_path))
    X_train, X_test, y_train, y_test = split_data_task(df)

    model = train_model_task(X_train, y_train)
    metrics = evaluate_model_task(model, X_test, y_test)
    logger.info(
        "Test metrics: %s", ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    )

    pickle_object_task(model, model_path)
    logger.info("Model saved to %s", model_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument(
        "trainset_path",
        type=Path,
        nargs="?",
        default=DATA_PATH,
        help="Path to the training set (default: data/abalone.csv)",
    )
    args = parser.parse_args()
    main(args.trainset_path)
