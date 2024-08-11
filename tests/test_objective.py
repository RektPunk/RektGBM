import pytest

from rektgbm.base import MethodName
from rektgbm.objective import OBJECTIVE_ENGINE_MAPPER, ObjectiveName, RektObjective
from rektgbm.task import TaskType


# Test RektObjective __post_init__ without objective (objective determined by task_type)
@pytest.mark.parametrize(
    "task_type, expected_objective",
    [
        (TaskType.regression, ObjectiveName.rmse),
        (TaskType.binary, ObjectiveName.binary),
        (TaskType.multiclass, ObjectiveName.multiclass),
        (TaskType.rank, ObjectiveName.lambdarank),
    ],
)
def test_rektobjective_post_init_without_objective(task_type, expected_objective):
    objective = RektObjective(task_type=task_type, objective=None)
    assert objective.objective == expected_objective


# Test RektObjective __post_init__ with objective provided
@pytest.mark.parametrize(
    "task_type, objective",
    [
        (TaskType.regression, "rmse"),
        (TaskType.binary, "binary"),
        (TaskType.multiclass, "multiclass"),
        (TaskType.rank, "lambdarank"),
    ],
)
def test_rektobjective_post_init_with_objective(task_type, objective):
    rekt_objective = RektObjective(task_type=task_type, objective=objective)
    assert rekt_objective.objective == ObjectiveName.get(objective)


# Test get_objective_str
@pytest.mark.parametrize(
    "method, expected_objective_str",
    [
        (
            MethodName.lightgbm,
            OBJECTIVE_ENGINE_MAPPER[ObjectiveName.rmse][MethodName.lightgbm],
        ),
        (
            MethodName.xgboost,
            OBJECTIVE_ENGINE_MAPPER[ObjectiveName.rmse][MethodName.xgboost],
        ),
    ],
)
def test_rektobjective_get_objective_str(method, expected_objective_str):
    objective = RektObjective(task_type=TaskType.regression, objective="rmse")
    assert objective.get_objective_str(method) == expected_objective_str


# Test get_objective_dict
@pytest.mark.parametrize(
    "method, expected_objective_dict",
    [
        (
            MethodName.lightgbm,
            {
                "objective": OBJECTIVE_ENGINE_MAPPER[ObjectiveName.rmse][
                    MethodName.lightgbm
                ]
            },
        ),
        (
            MethodName.xgboost,
            {
                "objective": OBJECTIVE_ENGINE_MAPPER[ObjectiveName.rmse][
                    MethodName.xgboost
                ]
            },
        ),
    ],
)
def test_rektobjective_get_objective_dict(method, expected_objective_dict):
    objective = RektObjective(task_type=TaskType.regression, objective="rmse")
    assert objective.get_objective_dict(method) == expected_objective_dict


# Test __validate_objective for valid objective
@pytest.mark.parametrize(
    "task_type, objective",
    [
        (TaskType.regression, ObjectiveName.rmse),
        (TaskType.binary, ObjectiveName.binary),
        (TaskType.multiclass, ObjectiveName.multiclass),
        (TaskType.rank, ObjectiveName.lambdarank),
    ],
)
def test_rektobjective_validate_objective_valid(task_type, objective):
    objective_instance = RektObjective(task_type=task_type, objective=objective.value)
    # If it doesn't raise an exception, it's correct
    assert objective_instance.objective == objective


# Test __validate_objective for invalid objective
def test_rektobjective_validate_objective_invalid():
    with pytest.raises(ValueError):
        RektObjective(task_type=TaskType.regression, objective="binary")
