from typing import Dict

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType


# TODO
# update objectives and metrics
# MAPPER: default
class ObjectiveName(BaseEnum):
    rmse: int = 1


class XgbObjectiveName(BaseEnum):
    squarederror: str = "reg:squarederror"
    logistic: str = "binary:logistic"
    softmax: str = "multi:softmax"


class XgbMetricName(BaseEnum):
    rmse: str = "rmse"
    loglosss: str = "logloss"
    mlogloss: str = "mlogloss"


XGB_OBJECTIVE_MAPPER: Dict[TaskType, str] = {
    TaskType.regression: XgbObjectiveName.squarederror.value,
    TaskType.binary: XgbObjectiveName.logistic.value,
    TaskType.multiclass: XgbObjectiveName.softmax.value,
}

XGB_METRIC_MAPPER: Dict[TaskType, str] = {
    TaskType.regression: XgbMetricName.rmse.value,
    TaskType.binary: XgbMetricName.loglosss.value,
    TaskType.multiclass: XgbMetricName.mlogloss.value,
}

LGB_OBJECTIVE_MAPPER: Dict[TaskType, str] = {
    TaskType.regression: "",
    TaskType.binary: "",
    TaskType.multiclass: "",
}

LGB_METRIC_MAPPER: Dict[TaskType, str] = {
    TaskType.regression: "",
    TaskType.binary: "",
    TaskType.multiclass: "",
}

## TODO
# create common objective mapper
# "rmse" -> "reg:squarederror", "rmse"
# create common eval mapper
OBJECTIVE_MAPPER: Dict[ObjectiveName, Dict[MethodName, str]] = {
    ObjectiveName.rmse: {
        MethodName.lightgbm: "rmse",
        MethodName.xgboost: XgbObjectiveName.squarederror.value,
    }
}

METRIC_MAPPER: Dict[ObjectiveName, Dict[MethodName, str]] = {
    ObjectiveName.rmse: {
        MethodName.lightgbm: "rmse",
        MethodName.xgboost: XgbMetricName.rmse.value,
    }
}
