from typing import Dict, List, Optional

from sklearn.utils.multiclass import type_of_target

from rektgbm.base import BaseEnum, YdataLike


class TaskType(BaseEnum):
    regression: str = "regression"
    binary: str = "binary"
    multiclass: str = "multiclass"
    rank: str = "rank"


class SklearnTaskType(BaseEnum):
    continuous: int = 1
    binary: int = 2
    multiclass: int = 3


SKLEARN_TASK_TYPE_MAPPER: Dict[SklearnTaskType, List[TaskType]] = {
    SklearnTaskType.continuous: [TaskType.regression],
    SklearnTaskType.binary: [TaskType.binary],
    SklearnTaskType.multiclass: [TaskType.multiclass, TaskType.rank],
}


def check_task_type(
    target: YdataLike,
    group: Optional[YdataLike],
    task_type: Optional[str],
) -> TaskType:
    if group is not None:
        return TaskType.rank

    _type_inferred: str = type_of_target(y=target)
    _sklearn_task_type = SklearnTaskType.get(_type_inferred)
    _task_types = SKLEARN_TASK_TYPE_MAPPER.get(_sklearn_task_type)
    if task_type is None:
        return _task_types[0]

    _user_defined_task_type = TaskType.get(task_type)
    if _user_defined_task_type not in _task_types:
        raise ValueError(
            "The inferred 'task_type' does not match the provided one. "
            f"Expected one of '{[_.value for _ in _task_types]}'."
        )
    return _user_defined_task_type
