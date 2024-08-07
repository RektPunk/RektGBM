from typing import Any, Dict, Optional

from rektgbm.base import BaseGBM, MethodName
from rektgbm.dataset import RektDataset
from rektgbm.engine import RektEngine
from rektgbm.metric import RektMetric
from rektgbm.objective import RektObjective
from rektgbm.task import check_task_type


class RektGBM(BaseGBM):
    def __init__(
        self,
        method: str,
        params: Dict[str, Any],
        task_type: Optional[str] = None,
        objective: Optional[str] = None,
        metric: Optional[str] = None,
    ):
        self.method = MethodName.get(method)
        self.params = params
        self.task_type = task_type
        self.objective = objective
        self.metric = metric

    def fit(
        self,
        dataset: RektDataset,
        valid_set: Optional[RektDataset] = None,
    ):
        self._task_type = check_task_type(
            target=dataset.label,
            task_type=self.task_type,
        )
        self.rekt_objective = RektObjective(
            task_type=self._task_type,
            objective=self.objective,
        )
        self.rekt_metric = RektMetric(
            task_type=self._task_type,
            metric=self.metric,
        )

        _objective = self.rekt_objective.get_objective(method=self.method)
        _metric = self.rekt_metric.get_metric(method=self.method)
        self.params.update({**_objective, **_metric})
        self.engine = RektEngine(
            method=self.method,
            params=self.params,
        )
        self.engine.fit(dataset=dataset, valid_set=valid_set)

    def predict(self, dataset: RektDataset):
        return self.engine.predict(dataset=dataset)
