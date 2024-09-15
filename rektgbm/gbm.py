from functools import cached_property

import numpy as np

from rektgbm.base import BaseGBM, MethodName, ParamsLike, StateException
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
        self._colnames = dataset.colnames
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
        self._is_fitted = True

    def predict(self, dataset: RektDataset):
        preds = self.engine.predict(dataset=dataset)
        if self._task_type in {TaskType.binary, TaskType.regression, TaskType.rank}:
            return preds

        if self.__is_lgb:
            preds = np.argmax(preds, axis=1).astype(int)
        else:
            preds = np.around(preds).astype(int)
        return self.label_encoder.inverse_transform(series=preds)

    @cached_property
    def feature_importance(self) -> np.ndarray:
        self.__check_fitted()
        importances = {str(k): 0 for k in self._colnames}
        if self.__is_lgb:
            _importance = self.engine.model.feature_importance(
                importance_type="gain"
            ).tolist()
            importances.update({str(k): v for k, v in zip(self._colnames, _importance)})
            return importances
        else:
            importances.update(self.engine.model.get_score(importance_type="gain"))
            return importances

    @property
    def __is_lgb(self) -> bool:
        return self.method == MethodName.lightgbm

    def __check_fitted(self):
        if not getattr(self, "_is_fitted", False):
            raise StateException("fit is not completed.")
