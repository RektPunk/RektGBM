from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rektgbm.base import MethodName, StateException
from rektgbm.dataset import RektDataset
from rektgbm.encoder import RektLabelEncoder
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


@pytest.fixture
def dummy_dataset():
    x_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=(100,))
    colnames = [f"feature_{i}" for i in range(x_train.shape[1])]
    x_train = pd.DataFrame(x_train, columns=colnames)
    dataset = RektDataset(x_train, y_train)
    return dataset


@pytest.fixture
def dummy_gbm_model():
    params = {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_estimators": 10,
    }
    model = RektGBM(method="lightgbm", params=params, task_type="binary")
    return model


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
    gbm.label_encoder = RektLabelEncoder()
    gbm.label_encoder.fit_label([0, 1, 2])

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


def test_feature_importance_before_fit_raises(dummy_gbm_model):
    with pytest.raises(StateException, match="fit is not completed"):
        _ = dummy_gbm_model.feature_importance


def test_feature_importance_after_fit(dummy_gbm_model, dummy_dataset):
    dummy_gbm_model.fit(dataset=dummy_dataset)
    feature_importances = dummy_gbm_model.feature_importance

    assert isinstance(
        feature_importances, dict
    ), "Feature importances should be a dictionary"
    assert len(feature_importances) == len(
        dummy_dataset.colnames
    ), "Feature importance length mismatch"
    for feature in dummy_dataset.colnames:
        assert (
            feature in feature_importances
        ), f"Feature {feature} not found in importance"


def test_feature_importance_nonzero(dummy_gbm_model, dummy_dataset):
    """Test that at least some feature importances are non-zero after training"""
    dummy_gbm_model.fit(dataset=dummy_dataset)
    feature_importances = dummy_gbm_model.feature_importance

    non_zero_importances = sum(
        importance > 0 for importance in feature_importances.values()
    )
    assert (
        non_zero_importances > 0
    ), "At least one feature should have non-zero importance"
