from unittest.mock import MagicMock

import numpy as np
import pytest

from rektgbm.base import MethodName
from rektgbm.metric import MetricName
from rektgbm.objective import ObjectiveName
from rektgbm.param import get_lgb_params, get_xgb_params, set_additional_params


@pytest.fixture
def mock_trial():
    trial = MagicMock()
    trial.suggest_float.side_effect = lambda name, low, high: (low + high) / 2
    trial.suggest_int.side_effect = lambda name, low, high: (low + high) // 2
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    return trial


def test_get_lgb_params(mock_trial):
    params = get_lgb_params(mock_trial)
    assert params["verbosity"] == -1
    np.testing.assert_allclose(params["learning_rate"], 0.505, rtol=1e-5)
    np.testing.assert_allclose(params["max_depth"], 5)
    np.testing.assert_allclose(params["lambda_l1"], 10.0, rtol=1e-5)
    np.testing.assert_allclose(params["lambda_l2"], 10.0, rtol=1e-5)
    np.testing.assert_allclose(params["num_leaves"], 129)
    np.testing.assert_allclose(params["feature_fraction"], 0.7, rtol=1e-5)
    np.testing.assert_allclose(params["bagging_fraction"], 0.7, rtol=1e-5)
    np.testing.assert_allclose(params["bagging_freq"], 4)
    np.testing.assert_allclose(params["n_estimators"], 7000)


def test_get_xgb_params(mock_trial):
    params = get_xgb_params(mock_trial)
    assert params["verbosity"] == 0
    np.testing.assert_allclose(params["learning_rate"], 0.505, rtol=1e-5)
    np.testing.assert_allclose(params["max_depth"], 5)
    np.testing.assert_allclose(params["reg_lambda"], 10.0, rtol=1e-5)
    np.testing.assert_allclose(params["reg_alpha"], 10.0, rtol=1e-5)
    np.testing.assert_allclose(params["subsample"], 0.55, rtol=1e-5)
    np.testing.assert_allclose(params["colsample_bytree"], 0.55, rtol=1e-5)
    np.testing.assert_allclose(params["n_estimators"], 7000)


@pytest.mark.parametrize(
    "objective, method, metric, num_class, expected",
    [
        (ObjectiveName.quantile, MethodName.lightgbm, "l2", None, {"alpha": 0.5}),
        (
            ObjectiveName.quantile,
            MethodName.xgboost,
            "l2",
            None,
            {"quantile_alpha": 0.5},
        ),
        (ObjectiveName.huber, MethodName.lightgbm, "l2", None, {"alpha": 0.5}),
        (ObjectiveName.huber, MethodName.xgboost, "l2", None, {"huber_slope": 0.5}),
        (ObjectiveName.multiclass, MethodName.lightgbm, "l2", 3, {"num_class": 3}),
        (
            ObjectiveName.lambdarank,
            MethodName.xgboost,
            MetricName.ndcg.value,
            None,
            {"eval_metric": "ndcg@10"},
        ),
        (
            ObjectiveName.ndcg,
            MethodName.xgboost,
            MetricName.map.value,
            None,
            {"eval_metric": "map@10"},
        ),
    ],
)
def test_set_additional_params(objective, method, metric, num_class, expected):
    params = {"quantile_alpha": 0.5, "huber_slope": 0.5, "eval_at": 10}
    updated_params = set_additional_params(params, objective, metric, method, num_class)

    for key, value in expected.items():
        if isinstance(value, (float, int)):
            np.testing.assert_allclose(updated_params.get(key), value, rtol=1e-5)
        else:
            assert updated_params.get(key) == value
