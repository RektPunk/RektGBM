from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.utils.multiclass import type_of_target

from rektgbm.base import BaseEnum


class TaskType(BaseEnum):
    regression: int = 1
    binary: int = 2
    multiclass: int = 3


class SklearnTaskType(BaseEnum):
    continuous: int = 1
    binary: int = 2
    multiclass: int = 3


SKLEARN_TASK_TYPE_MAPPER: Dict[SklearnTaskType, TaskType] = {
    SklearnTaskType.continuous: TaskType.regression,
    SklearnTaskType.binary: TaskType.binary,
    SklearnTaskType.multiclass: TaskType.multiclass,
}


def _convert_column_to_list(column: Union[str, List[str]]) -> List[str]:
    if isinstance(column, list) and len(column) != 1:
        raise ValueError(f"column must be str or list[str] of length 1")
    _column: List[str] = [column] if isinstance(column, str) else column
    return _column


def _check_column_in_data(data: pd.DataFrame, column: Union[str, List[str]]) -> None:
    _dcols = data.columns
    _missing_cols = [col for col in column if col not in _dcols]
    if _missing_cols:
        raise ValueError(f"The target is missing")


def _check_column(data: pd.DataFrame, column: Union[str, List[str]]) -> List[str]:
    _column = _convert_column_to_list(column=column)
    _check_column_in_data(data=data, column=column)
    return _column


def check_task_type(
    data: pd.DataFrame,
    target: Union[str, List[str]],
    task_type: Optional[str] = None,
) -> TaskType:
    _target: List[str] = _check_column(data=data, column=target)
    _type_inferred: str = type_of_target(data[_target].values)
    _sklearn_task_type = SklearnTaskType.get(_type_inferred)
    _task_type = SKLEARN_TASK_TYPE_MAPPER.get(_sklearn_task_type)
    if task_type is not None and _task_type != TaskType.get(task_type):
        raise ValueError(
            f"Unable to infer 'task_type'. Expected '{TaskType.get(task_type).value}', but got '{_task_type.value}'."
        )
    return _task_type
