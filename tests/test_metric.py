import pytest

from rektgbm.base import MethodName
from rektgbm.metric import METRIC_ENGINE_MAPPER, MetricName, RektMetric
from rektgbm.objective import ObjectiveName
from rektgbm.task import TaskType


# Test RektMetric __post_init__ without metric (metric determined by objective)
@pytest.mark.parametrize(
    "task_type, objective, expected_metric",
    [
        (TaskType.regression, ObjectiveName.rmse, MetricName.rmse),
        (TaskType.regression, ObjectiveName.mae, MetricName.mae),
        (TaskType.binary, ObjectiveName.binary, MetricName.logloss),
        (TaskType.multiclass, ObjectiveName.multiclass, MetricName.mlogloss),
        (TaskType.rank, ObjectiveName.lambdarank, MetricName.ndcg),
    ],
)
def test_rektmetric_post_init_without_metric(task_type, objective, expected_metric):
    metric = RektMetric(task_type=task_type, objective=objective, metric=None)
    assert metric.metric == expected_metric


# Test RektMetric __post_init__ with metric provided
@pytest.mark.parametrize(
    "task_type, objective, metric",
    [
        (TaskType.regression, ObjectiveName.rmse, "rmse"),
        (TaskType.binary, ObjectiveName.binary, "logloss"),
        (TaskType.multiclass, ObjectiveName.multiclass, "mlogloss"),
        (TaskType.rank, ObjectiveName.lambdarank, "ndcg"),
    ],
)
def test_rektmetric_post_init_with_metric(task_type, objective, metric):
    rekt_metric = RektMetric(task_type=task_type, objective=objective, metric=metric)
    assert rekt_metric.metric == MetricName.get(metric)


# Test get_metric_str
@pytest.mark.parametrize(
    "method, expected_metric_str",
    [
        (
            MethodName.lightgbm,
            METRIC_ENGINE_MAPPER[MetricName.rmse][MethodName.lightgbm],
        ),
        (MethodName.xgboost, METRIC_ENGINE_MAPPER[MetricName.rmse][MethodName.xgboost]),
    ],
)
def test_rektmetric_get_metric_str(method, expected_metric_str):
    metric = RektMetric(
        task_type=TaskType.regression, objective=ObjectiveName.rmse, metric="rmse"
    )
    assert metric.get_metric_str(method) == expected_metric_str


# Test get_metric_dict
@pytest.mark.parametrize(
    "method, expected_metric_dict",
    [
        (
            MethodName.lightgbm,
            {"metric": METRIC_ENGINE_MAPPER[MetricName.rmse][MethodName.lightgbm]},
        ),
        (
            MethodName.xgboost,
            {"eval_metric": METRIC_ENGINE_MAPPER[MetricName.rmse][MethodName.xgboost]},
        ),
    ],
)
def test_rektmetric_get_metric_dict(method, expected_metric_dict):
    metric = RektMetric(
        task_type=TaskType.regression, objective=ObjectiveName.rmse, metric="rmse"
    )
    assert metric.get_metric_dict(method) == expected_metric_dict


# Test __validate_metric for valid metric
@pytest.mark.parametrize(
    "task_type, metric",
    [
        (TaskType.regression, MetricName.rmse),
        (TaskType.binary, MetricName.logloss),
        (TaskType.multiclass, MetricName.mlogloss),
        (TaskType.rank, MetricName.ndcg),
    ],
)
def test_rektmetric_validate_metric_valid(task_type, metric):
    metric_instance = RektMetric(
        task_type=task_type, objective=ObjectiveName.rmse, metric=metric.value
    )
    # If it doesn't raise an exception, it's correct
    assert metric_instance.metric == metric


# Test __validate_metric for invalid metric
def test_rektmetric_validate_metric_invalid():
    with pytest.raises(ValueError):
        RektMetric(
            task_type=TaskType.regression, objective=ObjectiveName.rmse, metric="auc"
        )
