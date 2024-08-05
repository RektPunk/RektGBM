from typing import Any, Dict, List, Optional

from rektgbm.base import BaseEnum, BaseGBM, MethodName
from rektgbm.dataset import RektDataset
from rektgbm.engine import RektEngine
from rektgbm.metric import RektMetric
from rektgbm.objective import RektObjective
from rektgbm.task import check_task_type


class _RektMethods(BaseEnum):
    both: int = 1
    lightgbm: int = 2
    xgboost: int = 3


class RektGBM(BaseGBM):
    def __init__(
        self,
        method: str,
        params: Dict[str, Any],
        task_type: Optional[str] = None,
        objective: Optional[str] = None,
        metric: Optional[str] = None,
    ):
        if _RektMethods.both == _RektMethods.get(method):
            self.method = [MethodName.lightgbm, MethodName.xgboost]
        else:
            self.method = [MethodName.get(method)]
        self.params = params
        self.task_type = task_type
        self.objective = objective
        self.metric = metric

    def fit(
        self,
        dataset: RektDataset,
        valid_set: Optional[RektDataset] = None,
    ):
        self.task_type = check_task_type(
            target=dataset.label,
            task_type=self.task_type,
        )
        self.rekt_objective = RektObjective(
            task_type=self.task_type, objective=self.objective
        )
        self.rekt_metric = RektMetric(
            task_type=self.task_type,
            metric=self.metric,
        )
        _rekt_engines: List[RektEngine] = []
        for method in self.methods:
            _objective = self.rekt_objective.get_objective(method=method)
            _metric = self.rekt_metric.get_metric(method=method)
            _engine = RektEngine(
                params=self.params,
                objective=_objective,
                metric=_metric,
                method=method,
            )
            _engine.fit(dataset=dataset, valid_set=valid_set)
            _rekt_engines.append(_engine)

        # TODO: score comparison, choose best model
        self.engine = _engine
        self._fitted = True

    def predict(self, dataset: RektDataset):
        return self.engine.predict(dataset=dataset)
