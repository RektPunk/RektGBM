from dataclasses import dataclass

from rektgbm.base import BaseEnum, MethodName
from rektgbm.objective import ObjectiveName
from rektgbm.task import TaskType


class MetricName(BaseEnum):
    rmse: str = "rmse"
    mae: str = "mae"
    mape: str = "mape"
    huber: str = "huber"
    gamma: str = "gamma"
    gamma_deviance: str = "gamma_deviance"
    quantile: str = "quantile"
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
    quantile: str = "quantile"
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
    tweedie_nloglik: str = "tweedie-nloglik"
    aft_nloglik: str = "aft-nloglik"
    interval_regression_accuracy: str = "interval-regression-accuracy"


class LgbMetricName(BaseEnum):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric
    l1: str = "l1"
    l2: str = "l2"
    rmse: str = "rmse"
    quantile: str = "quantile"
    mape: str = "mape"
    huber: str = "huber"
    fair: str = "fair"
    poisson: str = "poisson"
    gamma: str = "gamma"
    gamma_deviance: str = "gamma_deviance"
    tweedie: str = "tweedie"
    ndcg: str = "ndcg"
    map: str = "map"
    auc: str = "auc"
    average_precision: str = "average_precision"
    binary_logloss: str = "binary_logloss"
    binary_error: str = "binary_error"
    auc_mu: str = "auc_mu"
    multi_logloss: str = "multi_logloss"
    multi_error: str = "multi_error"
    cross_entropy: str = "cross_entropy"
    cross_entropy_lambda: str = "cross_entropy_lambda"
    kullback_leibler: str = "kullback_leibler"


TASK_METRIC_MAPPER: dict[TaskType, list[MetricName]] = {
    TaskType.regression: [
        MetricName.rmse,
        MetricName.mae,
        MetricName.huber,
        MetricName.mape,
        MetricName.gamma,
        MetricName.gamma_deviance,
        MetricName.quantile,
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


OBJECTIVE_METRIC_MAPPER: dict[ObjectiveName, MetricName] = {
    ObjectiveName.rmse: MetricName.rmse,
    ObjectiveName.mae: MetricName.mae,
    ObjectiveName.huber: MetricName.huber,
    ObjectiveName.quantile: MetricName.quantile,
    ObjectiveName.gamma: MetricName.gamma,
    ObjectiveName.binary: MetricName.logloss,
    ObjectiveName.multiclass: MetricName.mlogloss,
    ObjectiveName.lambdarank: MetricName.ndcg,
    ObjectiveName.ndcg: MetricName.map,
}


METRIC_DICT_KEY_MAPPER: dict[MethodName, str] = {
    MethodName.lightgbm: "metric",
    MethodName.xgboost: "eval_metric",
}

METRIC_ENGINE_MAPPER: dict[MetricName, dict[MethodName, str]] = {
    MetricName.rmse: {
        MethodName.lightgbm: LgbMetricName.rmse.value,
        MethodName.xgboost: XgbMetricName.rmse.value,
    },
    MetricName.mae: {
        MethodName.lightgbm: LgbMetricName.l1.value,
        MethodName.xgboost: XgbMetricName.mae.value,
    },
    MetricName.huber: {
        MethodName.lightgbm: LgbMetricName.huber.value,
        MethodName.xgboost: XgbMetricName.mphe.value,
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
    MetricName.quantile: {
        MethodName.lightgbm: LgbMetricName.quantile.value,
        MethodName.xgboost: XgbMetricName.quantile.value,
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
    objective: ObjectiveName
    metric: str | None

    def __post_init__(self) -> None:
        if self.metric:
            self.metric = MetricName.get(self.metric)
            self.__validate_metric()
        else:
            self.metric = OBJECTIVE_METRIC_MAPPER.get(self.objective)

        self._metric_engine_mapper = METRIC_ENGINE_MAPPER.get(self.metric)

    def get_metric_str(self, method: MethodName) -> str:
        return self._metric_engine_mapper.get(method)

    def get_metric_dict(self, method: MethodName) -> dict[str, str]:
        return {METRIC_DICT_KEY_MAPPER.get(method): self.get_metric_str(method=method)}

    def __validate_metric(self) -> None:
        metrics = TASK_METRIC_MAPPER.get(self.task_type)
        if self.metric not in metrics:
            raise ValueError(
                f"Task type '{self.task_type}' and metric '{self.metric}' are not matched."
            )
