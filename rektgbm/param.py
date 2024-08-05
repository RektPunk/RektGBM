from typing import Dict, Union

from optuna import Trial

from rektgbm.base import MethodName


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
