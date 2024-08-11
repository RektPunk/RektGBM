from unittest.mock import MagicMock, patch

import pytest

from rektgbm.base import MethodName, StateException
from rektgbm.dataset import RektDataset
from rektgbm.optimizer import RektOptimizer


@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=RektDataset)
    dataset.label = [0, 1, 0, 1]
    dataset.group = None
    return dataset


@pytest.fixture
def mock_valid_set():
    valid_set = MagicMock(spec=RektDataset)
    valid_set.label = [0, 1]
    valid_set.group = None
    return valid_set


@patch("optuna.create_study", autospec=True)
def test_optimize_params_lightgbm(mock_create_study, mock_dataset, mock_valid_set):
    study = MagicMock()
    study.best_value = 0.1
    study.best_params = {"learning_rate": 0.05, "num_leaves": 31}
    mock_create_study.return_value = study

    optimizer = RektOptimizer(
        method="lightgbm",
        task_type="binary",
        objective="binary",
        metric="logloss",
        params=[
            lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)
            }
        ],
    )

    optimizer.optimize_params(
        dataset=mock_dataset, n_trials=10, valid_set=mock_valid_set
    )
    mock_create_study.assert_called_once()
    assert optimizer.studies[MethodName.lightgbm] == study


@patch("optuna.create_study", autospec=True)
def test_optimize_params_xgboost(mock_create_study, mock_dataset, mock_valid_set):
    study = MagicMock()
    study.best_value = 0.2
    study.best_params = {"learning_rate": 0.1, "max_depth": 6}
    mock_create_study.return_value = study

    optimizer = RektOptimizer(
        method="xgboost",
        task_type="binary",
        objective="binary",
        metric="logloss",
        params=[
            lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2)
            }
        ],
    )

    optimizer.optimize_params(
        dataset=mock_dataset, n_trials=10, valid_set=mock_valid_set
    )
    mock_create_study.assert_called_once()
    assert optimizer.studies[MethodName.xgboost] == study


@patch("rektgbm.optimizer.RektEngine", autospec=True)
@patch("optuna.create_study", autospec=True)
def test_optimize_params_both_methods(
    mock_create_study, mock_engine_class, mock_dataset, mock_valid_set
):
    study_lightgbm = MagicMock()
    study_lightgbm.best_value = 0.1
    study_lightgbm.best_params = {"learning_rate": 0.05, "num_leaves": 31}

    study_xgboost = MagicMock()
    study_xgboost.best_value = 0.2
    study_xgboost.best_params = {"learning_rate": 0.1, "max_depth": 6}

    mock_create_study.side_effect = [study_lightgbm, study_xgboost]

    optimizer = RektOptimizer(
        method="both",
        task_type="binary",
        objective="binary",
        metric="logloss",
        params=[
            lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)
            },
            lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2)
            },
        ],
    )

    optimizer.optimize_params(
        dataset=mock_dataset, n_trials=10, valid_set=mock_valid_set
    )

    mock_create_study.assert_called()
    assert optimizer.studies[MethodName.lightgbm] == study_lightgbm
    assert optimizer.studies[MethodName.xgboost] == study_xgboost


def test_best_params():
    optimizer = RektOptimizer(
        method="lightgbm",
        task_type="binary",
        objective="binary",
        metric="logloss",
    )

    with pytest.raises(StateException):
        optimizer.best_params


@patch("rektgbm.optimizer.RektEngine", autospec=True)
@patch("optuna.create_study", autospec=True)
def test_best_params_after_optimization(
    mock_create_study, mock_engine_class, mock_dataset, mock_valid_set
):
    mock_engine = MagicMock()
    mock_engine.eval_loss = 0.1
    mock_engine_class.return_value = mock_engine

    study = MagicMock()
    study.best_value = 0.1
    study.best_params = {"learning_rate": 0.05, "num_leaves": 31}
    mock_create_study.return_value = study

    optimizer = RektOptimizer(
        method="lightgbm",
        task_type="binary",
        objective="binary",
        metric="logloss",
        params=[
            lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)
            }
        ],
    )

    optimizer.optimize_params(
        dataset=mock_dataset, n_trials=10, valid_set=mock_valid_set
    )
    best_params = optimizer.best_params

    assert best_params["method"] == "lightgbm"
    assert best_params["params"]["learning_rate"] == 0.05
    assert best_params["params"]["num_leaves"] == 31
    assert best_params["task_type"] == "binary"
    assert best_params["objective"] == "binary"
    assert best_params["metric"] == "logloss"


def test_optimize_params_without_valid_set_raises_value_error(mock_dataset):
    optimizer = RektOptimizer(
        method="lightgbm",
        task_type="rank",
        objective="rank",
        metric="ndcg",
    )
    with pytest.raises(ValueError):
        optimizer.optimize_params(dataset=mock_dataset, n_trials=10)
