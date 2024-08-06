from dataclasses import dataclass
from typing import Dict, List, Optional

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType

# TODO
# 1. add alias
# 2. TASK_OBJECTIVE_MAPPER: [0]: default
# 3. update objectives and metrics

## TODO
# create common objective mapper
# "rmse" -> "reg:squarederror", "rmse"
OBJECTIVE_DICT_KEY: str = "objective"


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


OBJECTIVE_ENGINE_MAPPER: Dict[ObjectiveName, Dict[MethodName, str]] = {
    ObjectiveName.rmse: {
        MethodName.lightgbm: LgbObjectiveName.regression.value,
        MethodName.xgboost: XgbObjectiveName.squarederror.value,
    }
}


@dataclass
class RektObjective:
    task_type: TaskType
    objective: Optional[str]

    def __post_init__(self) -> None:
        if self.objective:
            self.objective = ObjectiveName.get(self.objective)
            self.__validate_objective()
        else:
            _objectives = TASK_OBJECTIVE_MAPPER.get(self.task_type)
            self.objective = _objectives[0]

        self._objective_engine_mapper = OBJECTIVE_ENGINE_MAPPER.get(self.objective)

    def get_objective_str(self, method: MethodName) -> str:
        return self._objective_engine_mapper.get(method)

    def get_objective(self, method: MethodName) -> Dict[str, str]:
        return {OBJECTIVE_DICT_KEY: self.get_objective_str(method=method)}

    def __validate_objective(self) -> None:
        objectives = TASK_OBJECTIVE_MAPPER.get(self.task_type)
        if self.objective not in objectives:
            raise ValueError(
                f"Task type '{self.task_type}' and objective '{self.objective}' are not matched."
            )
