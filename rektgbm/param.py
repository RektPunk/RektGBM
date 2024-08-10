from typing import Any, Dict, Optional, Union

from optuna import Trial

from rektgbm.base import MethodName
from rektgbm.metric import MetricName
from rektgbm.objective import ObjectiveName


def get_lgb_params(trial: Trial) -> Dict[str, Union[float, int]]:
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#learning-control-parameters
    return {
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 20.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 20.0),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "n_estimators": trial.suggest_categorical("n_estimators", [7000, 15000, 20000]),
    }


def get_xgb_params(trial: Trial) -> Dict[str, Union[float, int]]:
    # https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
    return {
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "n_estimators": trial.suggest_categorical("n_estimators", [7000, 15000, 20000]),
    }


METHOD_PARAMS_MAPPER = {
    MethodName.lightgbm: get_lgb_params,
    MethodName.xgboost: get_xgb_params,
}


def set_additional_params(
    params: Dict[str, Any],
    objective: ObjectiveName,
    metric: str,
    method: MethodName,
    num_class: Optional[int],
) -> Dict[str, Any]:
    _params = params.copy()
    if objective == ObjectiveName.quantile:
        if method == MethodName.lightgbm and "quantile_alpha" in _params.keys():
            _params["alpha"] = _params.pop("quantile_alpha")
        elif method == MethodName.xgboost and "alpha" in _params.keys():
            _params["quantile_alpha"] = _params.pop("alpha")
    elif objective == ObjectiveName.huber:
        if method == MethodName.lightgbm and "huber_slope" in _params.keys():
            _params["alpha"] = _params.pop("quantile_alpha")
        elif method == MethodName.xgboost and "alpha" in _params.keys():
            _params["huber_slope"] = _params.pop("alpha")
    elif objective == ObjectiveName.multiclass:
        _params["num_class"] = num_class

    if metric in {MetricName.ndcg.value, MetricName.map.value}:
        _eval_at_defalut: int = 5
        _eval_at = _params.pop("eval_at", _eval_at_defalut)
        if method == MethodName.xgboost:
            _params["eval_metric"] = f"{metric}@{_eval_at}"
        elif method == MethodName.lightgbm:
            _params["eval_at"] = _eval_at
    return _params
