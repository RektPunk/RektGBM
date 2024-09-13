from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rektgbm.base import MethodName
from rektgbm.dataset import RektDataset
from rektgbm.engine import RektEngine
from rektgbm.gbm import RektGBM
from rektgbm.task import TaskType


@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=RektDataset)
    dataset.label = np.array([0, 1, 0, 1])
    dataset.group = None
    dataset.dsplit.return_value = ("dtrain_mock", "dvalid_mock")
    dataset.dtrain.return_value = "dtrain_mock"
    dataset.dpredict.return_value = np.array([0.1, 0.2, 0.3])
    dataset.fit_transform_label.return_value = np.array([0, 1, 0, 1])
    dataset.transform_label.return_value = None
    return dataset


@pytest.fixture
def mock_valid_set():
    valid_set = MagicMock(spec=RektDataset)
    valid_set.label = np.array([0, 1])
    valid_set.group = None
    valid_set.transform_label.return_value = None
    return valid_set


@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=RektEngine)
    engine.predict.return_value = np.array([0.1, 0.9, 0.2, 0.8])
    return engine


@patch("rektgbm.gbm.RektEngine", autospec=True)
def test_rektgbm_fit(mock_engine_class, mock_dataset, mock_valid_set, mock_engine):
    mock_engine_class.return_value = mock_engine

    gbm = RektGBM(
        method="lightgbm",
        params={},
        task_type="binary",
        objective="binary",
        metric="logloss",
    )

    gbm.fit(dataset=mock_dataset, valid_set=mock_valid_set)
    assert gbm._task_type == TaskType.binary
    mock_engine_class.assert_called_once_with(
        method=MethodName.lightgbm,
        params={"metric": "binary_logloss", "objective": "binary"},
        task_type=TaskType.binary,
    )
    mock_engine.fit.assert_called_once_with(
        dataset=mock_dataset, valid_set=mock_valid_set
    )


def test_rektgbm_predict_binary(mock_dataset, mock_engine):
    gbm = RektGBM(
        method="lightgbm",
        params={"metric": "l2"},
        task_type="binary",
        objective="binary",
        metric="binary_error",
    )
    gbm.engine = mock_engine
    gbm._task_type = TaskType.binary
    gbm._is_fitted = True

    preds = gbm.predict(dataset=mock_dataset)
    np.testing.assert_allclose(preds, [0.1, 0.9, 0.2, 0.8], rtol=1e-5)


def test_rektgbm_predict_multiclass(mock_dataset, mock_engine):
    gbm = RektGBM(
        method="lightgbm",
        params={"metric": "l2"},
        task_type="multiclass",
        objective="multiclass",
        metric="multi_logloss",
    )
    gbm.engine = mock_engine
    gbm._task_type = TaskType.multiclass
    gbm._is_fitted = True

    mock_engine.predict.return_value = np.array(
        [[0.1, 0.7, 0.2], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6]]
    )

    preds = gbm.predict(dataset=mock_dataset)
    np.testing.assert_allclose(preds, [1, 1, 2], rtol=1e-5)


def test_rektgbm_predict_rank(mock_dataset, mock_engine):
    gbm = RektGBM(
        method="lightgbm",
        params={"metric": "l2"},
        task_type="rank",
        objective="rank",
        metric="ndcg",
    )
    gbm.engine = mock_engine
    gbm._task_type = TaskType.rank
    gbm._is_fitted = True

    preds = gbm.predict(dataset=mock_dataset)
    np.testing.assert_allclose(preds, [0.1, 0.9, 0.2, 0.8], rtol=1e-5)


def test_rektgbm_predict_unfitted_error(mock_dataset):
    gbm = RektGBM(
        method="lightgbm",
        params={"metric": "l2"},
    )
    with pytest.raises(AttributeError):
        gbm.predict(dataset=mock_dataset)


def test_rektgbm_fit_rank_raises_value_error(mock_dataset):
    gbm = RektGBM(
        method="lightgbm",
        params={"metric": "l2"},
        task_type="rank",
    )
    with pytest.raises(ValueError):
        gbm.fit(dataset=mock_dataset)
