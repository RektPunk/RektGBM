from typing import Dict, List

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType

# TODO
# 1. add alias
# 2. TASK_METRIC_MAPPER: [0]: default
# 3. update objectives and metrics


class MetricName(BaseEnum):
    rmse: int = 1
    logloss: int = 2
    mlogloss: int = 3


class XgbMetricName(BaseEnum):
    # https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
    rmse: str = "rmse"
    loglosss: str = "logloss"
    mlogloss: str = "mlogloss"


class LgbMetricName(BaseEnum):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric
    rmse: str = "rmse"
    binary_logloss: str = "binary_logloss"
    multi_logloss: str = "multi_logloss"


TASK_OBJECTIVE_MAPPER: Dict[TaskType, List[MetricName]] = {
    TaskType.regression: [MetricName.rmse],
    TaskType.binary: [MetricName.logloss],
    TaskType.multiclass: [MetricName.mlogloss],
}


## TODO
# create common objective mapper
# "rmse" -> "reg:squarederror", "rmse"
# create common eval mapper

METRIC_ENGINE_STR_MAPPER: Dict[MetricName, Dict[MethodName, str]] = {
    MetricName.rmse: {
        MethodName.lightgbm: LgbMetricName.rmse.value,
        MethodName.xgboost: XgbMetricName.rmse.value,
    }
}
