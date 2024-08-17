import logging
from pprint import pformat

import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

from rektgbm import RektDataset, RektGBM, RektOptimizer

logging.basicConfig(level=logging.INFO)


def read_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)


def save_data(preds: np.ndarray, file_path: str) -> None:
    if file_path.endswith(".csv"):
        pd.DataFrame(preds, columns=["predict"]).to_csv(file_path, index=False)


def main(
    data_path: Annotated[str, typer.Argument(help="Path to the training data file.")],
    test_data_path: Annotated[str, typer.Argument(help="Path to the test data file.")],
    target: Annotated[str, typer.Argument(help="Name of the target column.")],
    result_path: Annotated[
        str,
        typer.Argument(help="Path to save the prediction results."),
    ] = "predict.csv",
    n_trials: Annotated[
        int, typer.Argument(help="Number of optimization trials.")
    ] = 100,
) -> None:
    if (
        not data_path.endswith(".csv")
        or not test_data_path.endswith(".csv")
        or not result_path.endswith(".csv")
    ):
        raise ValueError("Unsupported file format. Please provide a CSV file.")

    train_data = read_data(data_path)
    test_data = read_data(test_data_path)
    train_label = train_data.pop(target)
    dtrain = RektDataset(data=train_data, label=train_label)
    dtest = RektDataset(data=test_data)
    rekt_optimizer = RektOptimizer()

    rekt_optimizer.optimize_params(dataset=dtrain, n_trials=n_trials)
    logging.info("Best params:\n%s", pformat(rekt_optimizer.best_params))
    rekt_gbm = RektGBM(**rekt_optimizer.best_params)
    rekt_gbm.fit(dataset=dtrain)
    preds = rekt_gbm.predict(dataset=dtest)

    save_data(preds, result_path)


def run():
    typer.run(main)
