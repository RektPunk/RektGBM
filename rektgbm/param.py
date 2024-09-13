from optuna import Trial

from rektgbm.base import MethodName, ParamsLike
from rektgbm.metric import MetricName
from rektgbm.objective import ObjectiveName


def get_lgb_params(trial: Trial) -> ParamsLike:
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
    }


def get_xgb_params(trial: Trial) -> ParamsLike:
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
    params: ParamsLike,
    objective: ObjectiveName,
    metric: str,
    method: MethodName,
    num_class: int | None,
) -> ParamsLike:
    _params = params.copy()

    def _handle_quantile_params():
        if method == MethodName.lightgbm:
            _params["alpha"] = _params.pop("quantile_alpha", _params.get("alpha"))
        elif method == MethodName.xgboost:
            _params["quantile_alpha"] = _params.pop(
                "alpha", _params.get("quantile_alpha")
            )

    def _handle_huber_params():
        if method == MethodName.lightgbm:
            _params["alpha"] = _params.pop("quantile_alpha", _params.get("alpha"))
        elif method == MethodName.xgboost:
            _params["huber_slope"] = _params.pop("alpha", _params.get("huber_slope"))

    def _handle_multiclass_params():
        if num_class is not None:
            _params["num_class"] = num_class

    def _handle_rank_metric_params():
        eval_at_default = 5
        eval_at = _params.pop("eval_at", eval_at_default)
        if method == MethodName.xgboost:
            _params["eval_metric"] = f"{metric}@{eval_at}"
        elif method == MethodName.lightgbm:
            _params["eval_at"] = eval_at

    objective_handler_map = {
        ObjectiveName.quantile: _handle_quantile_params,
        ObjectiveName.huber: _handle_huber_params,
        ObjectiveName.multiclass: _handle_multiclass_params,
    }

    if objective in objective_handler_map:
        objective_handler_map[objective]()

    if metric in {MetricName.ndcg.value, MetricName.map.value}:
        _handle_rank_metric_params()

    return _params
