import pytest
from sklearn.datasets import make_classification, make_regression

from rektgbm.task import TaskType, check_task_type

regression_data, regression_label = make_regression(n_samples=100, n_features=10)
binary_data, binary_label = make_classification(
    n_samples=100, n_features=10, n_classes=2
)
multiclass_data, multiclass_label = make_classification(
    n_samples=100, n_features=10, n_informative=5, n_classes=3
)
group = [1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize(
    "target, group, task_type, expected",
    [
        (regression_label, None, None, TaskType.regression),  # Regression task
        (binary_label, None, None, TaskType.binary),  # Binary classification task
        (
            multiclass_label,
            None,
            None,
            TaskType.multiclass,
        ),  # Multiclass classification task
        (multiclass_label, group, None, TaskType.rank),  # Rank task with group
        (
            binary_label,
            None,
            "binary",
            TaskType.binary,
        ),  # User-defined binary classification task
    ],
)
def test_check_task_type(target, group, task_type, expected):
    inferred_task_type = check_task_type(
        target=target, group=group, task_type=task_type
    )
    assert inferred_task_type == expected


@pytest.mark.parametrize(
    "target, task_type",
    [
        (
            binary_label,
            "regression",
        ),  # Mismatch between inferred binary and user-defined regression
    ],
)
def test_check_task_type_user_defined_mismatch(target, task_type):
    with pytest.raises(
        ValueError, match="The inferred 'task_type' does not match the provided one."
    ):
        check_task_type(target=target, group=None, task_type=task_type)
