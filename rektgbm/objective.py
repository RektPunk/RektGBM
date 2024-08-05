from typing import Dict, List

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType


# TODO
# 1. add alias
# 2. TASK_OBJECTIVE_MAPPER: [0]: default
# 3. update objectives and metrics
class ObjectiveName(BaseEnum):
    rmse: int = 1
    binary: int = 2
    multiclass: int = 3


class XgbObjectiveName(BaseEnum):
    # https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
    squarederror: str = "reg:squarederror"
    logistic: str = "binary:logistic"
    softmax: str = "multi:softmax"


class LgbObjectiveName(BaseEnum):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters
    regression: str = "regression"
    binary: str = "binary"
    multiclass: str = "multiclass"


TASK_OBJECTIVE_MAPPER: Dict[TaskType, List[ObjectiveName]] = {
    TaskType.regression: [ObjectiveName.rmse],
    TaskType.binary: [ObjectiveName.binary],
    TaskType.multiclass: [ObjectiveName.multiclass],
}

## TODO
# create common objective mapper
# "rmse" -> "reg:squarederror", "rmse"
# create common eval mapper
OBJECTIVE_ENGINE_STR_MAPPER: Dict[ObjectiveName, Dict[MethodName, str]] = {
    ObjectiveName.rmse: {
        MethodName.lightgbm: LgbObjectiveName.regression.value,
        MethodName.xgboost: XgbObjectiveName.squarederror.value,
    }
}
