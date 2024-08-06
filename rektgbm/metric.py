from dataclasses import dataclass
from typing import Dict, List, Optional

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType

# TODO
# 1. add alias
# 2. TASK_METRIC_MAPPER: [0]: default
# 3. update objectives and metrics

## TODO
# create common eval mapper
# "rmse" -> "reg:squarederror", "rmse"


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


TASK_METRIC_MAPPER: Dict[TaskType, List[MetricName]] = {
    TaskType.regression: [MetricName.rmse],
    TaskType.binary: [MetricName.logloss],
    TaskType.multiclass: [MetricName.mlogloss],
}


METRIC_DICT_KEY_MAPPER: Dict[MethodName, str] = {
    MethodName.lightgbm: "metric",
    MethodName.xgboost: "eval_metric",
}

METRIC_ENGINE_MAPPER: Dict[MetricName, Dict[MethodName, str]] = {
    MetricName.rmse: {
        MethodName.lightgbm: LgbMetricName.rmse.value,
        MethodName.xgboost: XgbMetricName.rmse.value,
    }
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
