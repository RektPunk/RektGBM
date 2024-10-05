from unittest.mock import MagicMock

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

from rektgbm.base import MethodName, StateException
from rektgbm.dataset import RektDataset
from rektgbm.engine import _VALID_STR, RektEngine
from rektgbm.task import TaskType


@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=RektDataset)
    dataset.dsplit.return_value = ("dtrain_mock", "dvalid_mock")
    dataset.dtrain.return_value = "dtrain_mock"
    dataset.dpredict.return_value = np.array([0.1, 0.2, 0.3])
    return dataset


@pytest.fixture
def mock_lgb_model():
    model = MagicMock()
    model.predict.return_value = np.array([0.1, 0.2, 0.3])
    model.best_score = {_VALID_STR: {"l2": 0.05}}
    return model


@pytest.fixture
def mock_xgb_model():
    model = MagicMock()
    model.predict.return_value = np.array([0.1, 0.2, 0.3])
    return model


def test_rektengine_lgb_fit(mock_dataset, mock_lgb_model):
    engine = RektEngine(
        method=MethodName.lightgbm,
        params={"metric": "l2"},
        task_type=TaskType.regression,
    )
    lgb.train = MagicMock(return_value=mock_lgb_model)
    engine.fit(dataset=mock_dataset, valid_set=None)

    assert engine._is_fitted
    lgb.train.assert_called_once_with(
        train_set="dtrain_mock",
        params={"metric": "l2"},
        valid_sets="dvalid_mock",
        valid_names=_VALID_STR,
    )


def test_rektengine_xgb_fit(mock_dataset, mock_xgb_model):
    engine = RektEngine(
        method=MethodName.xgboost,
        params={"eval_metric": "rmse"},
        task_type=TaskType.regression,
    )
    xgb.train = MagicMock(return_value=mock_xgb_model)
    engine.fit(dataset=mock_dataset, valid_set=None)

    assert engine._is_fitted
    xgb.train.assert_called_once_with(
        dtrain="dtrain_mock",
        verbose_eval=False,
        num_boost_round=100,
        params={"eval_metric": "rmse"},
        evals_result={},
        evals=[("dvalid_mock", _VALID_STR)],
    )


def test_rektengine_predict(mock_dataset, mock_lgb_model):
    engine = RektEngine(
        method=MethodName.lightgbm,
        params={"metric": "l2"},
        task_type=TaskType.regression,
    )
    engine.model = mock_lgb_model
    engine._is_fitted = True

    pred = engine.predict(dataset=mock_dataset)
    np.testing.assert_allclose(pred, [0.1, 0.2, 0.3], rtol=1e-5)


def test_rektengine_eval_loss_lgb(mock_lgb_model):
    engine = RektEngine(
        method=MethodName.lightgbm,
        params={"metric": "l2"},
        task_type=TaskType.regression,
    )
    engine.model = mock_lgb_model
    engine._is_fitted = True

    loss = engine.eval_loss
    np.testing.assert_allclose(loss, 0.05, rtol=1e-5)


def test_rektengine_eval_loss_xgb(mock_xgb_model):
    engine = RektEngine(
        method=MethodName.xgboost,
        params={"eval_metric": "rmse"},
        task_type=TaskType.regression,
    )
    engine.model = mock_xgb_model
    engine._is_fitted = True
    engine.evals_result = {_VALID_STR: {"rmse": [0.1, 0.05]}}

    loss = engine.eval_loss
    np.testing.assert_allclose(loss, 0.05, rtol=1e-5)


def test_rektengine_not_fitted_error(mock_dataset):
    engine = RektEngine(
        method=MethodName.lightgbm,
        params={"metric": "l2"},
        task_type=TaskType.regression,
    )

    with pytest.raises(StateException):
        engine.predict(dataset=mock_dataset)

    with pytest.raises(StateException):
        _ = engine.eval_loss
