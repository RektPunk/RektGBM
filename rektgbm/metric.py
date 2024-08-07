from dataclasses import dataclass
from typing import Dict, List, Optional

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType


class MetricName(BaseEnum):
    rmse: str = "rmse"
    mae: str = "mae"
    mse: str = "mse"
    mape: str = "mape"
    gamma: str = "gamma"
    gamma_deviance: str = "gamma_deviance"
    poisson: str = "poisson"
    tweedie: str = "tweedie"
    logloss: str = "logloss"
    auc: str = "auc"
    mlogloss: str = "mlogloss"
    ndcg: str = "ndcg"
    map: str = "map"


class XgbMetricName(BaseEnum):
    # https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
    rmse: str = "rmse"
    rmsle: str = "rmsle"
    mae: str = "mae"
    mape: str = "mape"
    mphe: str = "mphe"
    logloss: str = "logloss"
    error: str = "error"
    merror: str = "merror"
    mlogloss: str = "mlogloss"
    auc: str = "auc"
    aucpr: str = "aucpr"
    ndcg: str = "ndcg"
    map: str = "map"
    cox_nloglik: str = "cox-nloglik"
    gamma_nloglik: str = "gamma-nloglik"
    gamma_deviance: str = "gamma-deviance"
    poisson_nloglik: str = "poisson-nloglik"
    poisson_deviance: str = "poisson-deviance"
    tweedie_nloglik: str = "tweedie-nloglik"
    aft_nloglik: str = "aft-nloglik"
    interval_regression_accuracy: str = "interval-regression-accuracy"


class LgbMetricName(BaseEnum):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric
    mae: str = "mae"
    mse: str = "mse"
    rmse: str = "rmse"
    quantile: str = "quantile"
    mape: str = "mape"
    binary_logloss: str = "binary_logloss"
    binary_error: str = "binary_error"
    multi_logloss: str = "multi_logloss"
    multi_error: str = "multi_error"
    huber: str = "huber"
    fair: str = "fair"
    poisson: str = "poisson"
    gamma: str = "gamma"
    gamma_deviance: str = "gamma_deviance"
    tweedie: str = "tweedie"
    ndcg: str = "ndcg"
    lambdarank: str = "ndcg"
    cross_entropy: str = "cross_entropy"
    cross_entropy_lambda: str = "cross_entropy_lambda"
    kullback_leibler: str = "kullback_leibler"
    map: str = "map"
    mean_average_precision: str = "mean_average_precision"
    auc: str = "auc"
    average_precision: str = "average_precision"
    auc_mu: str = "auc_mu"


TASK_METRIC_MAPPER: Dict[TaskType, List[MetricName]] = {
    TaskType.regression: [
        MetricName.rmse,
        MetricName.mae,
        MetricName.mse,
        MetricName.mape,
        MetricName.gamma,
        MetricName.gamma_deviance,
        MetricName.poisson,
        MetricName.tweedie,
    ],
    TaskType.binary: [
        MetricName.logloss,
        MetricName.auc,
    ],
    TaskType.multiclass: [
        MetricName.mlogloss,
    ],
    TaskType.rank: [
        MetricName.ndcg,
        MetricName.map,
    ],
}


METRIC_DICT_KEY_MAPPER: Dict[MethodName, str] = {
    MethodName.lightgbm: "metric",
    MethodName.xgboost: "eval_metric",
}

METRIC_ENGINE_MAPPER: Dict[MetricName, Dict[MethodName, str]] = {
    MetricName.rmse: {
        MethodName.lightgbm: LgbMetricName.rmse.value,
        MethodName.xgboost: XgbMetricName.rmse.value,
    },
    MetricName.mae: {
        MethodName.lightgbm: LgbMetricName.mae.value,
        MethodName.xgboost: XgbMetricName.mae.value,
    },
    MetricName.logloss: {
        MethodName.lightgbm: LgbMetricName.binary_logloss.value,
        MethodName.xgboost: XgbMetricName.logloss.value,
    },
    MetricName.mlogloss: {
        MethodName.lightgbm: LgbMetricName.multi_logloss.value,
        MethodName.xgboost: XgbMetricName.mlogloss.value,
    },
    MetricName.auc: {
        MethodName.lightgbm: LgbMetricName.auc.value,
        MethodName.xgboost: XgbMetricName.auc.value,
    },
    MetricName.mape: {
        MethodName.lightgbm: LgbMetricName.mape.value,
        MethodName.xgboost: XgbMetricName.mape.value,
    },
    MetricName.gamma: {
        MethodName.lightgbm: LgbMetricName.gamma.value,
        MethodName.xgboost: XgbMetricName.gamma_nloglik.value,
    },
    MetricName.gamma_deviance: {
        MethodName.lightgbm: LgbMetricName.gamma_deviance.value,
        MethodName.xgboost: XgbMetricName.gamma_deviance.value,
    },
    MetricName.poisson: {
        MethodName.lightgbm: LgbMetricName.poisson.value,
        MethodName.xgboost: XgbMetricName.poisson_nloglik.value,
    },
    MetricName.tweedie: {
        MethodName.lightgbm: LgbMetricName.tweedie.value,
        MethodName.xgboost: XgbMetricName.tweedie_nloglik.value,
    },
    MetricName.ndcg: {
        MethodName.lightgbm: LgbMetricName.ndcg.value,
        MethodName.xgboost: XgbMetricName.ndcg.value,
    },
    MetricName.map: {
        MethodName.lightgbm: LgbMetricName.map.value,
        MethodName.xgboost: XgbMetricName.map.value,
    },
}


@dataclass
class RektMetric:
    task_type: TaskType
    metric: Optional[str]

    def __post_init__(self) -> None:
        if self.metric:
            self.metric = MetricName.get(self.metric)
            self.__validate_metric()
        else:
            _metrics = TASK_METRIC_MAPPER.get(self.task_type)
            self.metric = _metrics[0]

        self._metric_engine_mapper = METRIC_ENGINE_MAPPER.get(self.metric)

    def get_metric_str(self, method: MethodName) -> str:
        return self._metric_engine_mapper.get(method)

    def get_metric(self, method: MethodName) -> Dict[str, str]:
        return {METRIC_DICT_KEY_MAPPER.get(method): self.get_metric_str(method=method)}

    def __validate_metric(self) -> None:
        metrics = TASK_METRIC_MAPPER.get(self.task_type)
        if self.metric not in metrics:
            raise ValueError(
                f"Task type '{self.task_type}' and metric '{self.metric}' are not matched."
            )
