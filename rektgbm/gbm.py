import numpy as np

from rektgbm.base import BaseGBM, MethodName, ParamsLike
from rektgbm.dataset import RektDataset
from rektgbm.engine import RektEngine
from rektgbm.metric import RektMetric
from rektgbm.objective import RektObjective
from rektgbm.task import TaskType, check_task_type


class RektGBM(BaseGBM):
    def __init__(
        self,
        method: str,
        params: ParamsLike,
        task_type: str | None = None,
        objective: str | None = None,
        metric: str | None = None,
    ):
        self.method = MethodName.get(method)
        self.params = params
        self.task_type = task_type
        self.objective = objective
        self.metric = metric

    def fit(
        self,
        dataset: RektDataset,
        valid_set: RektDataset | None = None,
    ):
        self._task_type = check_task_type(
            target=dataset.label,
            group=dataset.group,
            task_type=self.task_type,
        )
        if self._task_type == TaskType.rank and valid_set is None:
            raise ValueError(
                "A validation set must be provided when using the 'rank' task."
            )

        self.rekt_objective = RektObjective(
            task_type=self._task_type,
            objective=self.objective,
        )
        self.rekt_metric = RektMetric(
            task_type=self._task_type,
            objective=self.rekt_objective.objective,
            metric=self.metric,
        )
        if self._task_type in {TaskType.binary, TaskType.multiclass, TaskType.rank}:
            self.label_encoder = dataset.fit_transform_label()
            if valid_set:
                valid_set.transform_label(label_encoder=self.label_encoder)

        _objective = self.rekt_objective.get_objective_dict(method=self.method)
        _metric = self.rekt_metric.get_metric_dict(method=self.method)
        self.params.update({**_objective, **_metric})
        self.engine = RektEngine(
            method=self.method,
            params=self.params,
            task_type=self._task_type,
        )
        self.engine.fit(dataset=dataset, valid_set=valid_set)

    def predict(self, dataset: RektDataset):
        preds = self.engine.predict(dataset=dataset)
        if self._task_type in {TaskType.binary, TaskType.regression, TaskType.rank}:
            return preds

        if self.method == MethodName.lightgbm:
            preds = np.argmax(preds, axis=1).astype(int)
        else:
            preds = np.around(preds).astype(int)
        return self.label_encoder.inverse_transform(series=preds)
