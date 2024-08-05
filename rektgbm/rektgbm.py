from typing import Any, Dict, Optional

from rektgbm.base import BaseGBM, MethodName, RektException
from rektgbm.dataset import RektDataset
from rektgbm.engine import Engine
from rektgbm.task import check_task_type


class RektGBM(BaseGBM):
    # TODO:
    # rektgbm -- select lightgbm, xgboost, or both
    # fit -> predict
    def __init__(
        self,
        method: str = "both",
        params: Optional[Dict[str, Any]] = None,
    ):
        if method == "both":
            self.method = [MethodName.lightgbm, MethodName.xgboost]
        else:
            self.method = [MethodName.get(method)]
        self.params = params

    def fit(
        self,
        dataset: RektDataset,
        valid_set: Optional[RektDataset] = None,
        objective: Optional[str] = None,
        metric: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        self.task_type = check_task_type(
            target=dataset.label,
            task_type=task_type,
        )
        self.objective = objective  ##FIXME
        self.metric = metric  ##FIXME

        for method in self.methods:
            _engine = Engine(
                params=self.params,
                objective=self.objective,
                metric=self.metric,
                method=method,
            )
            _engine.fit(dataset=dataset, valid_set=valid_set)

        # TODO: score comparison, choose best model
        self.engine = _engine
        self._is_fitted = True

    def predict(self, dataset: RektDataset):
        self.__predict_available()
        return self.engine.predict(dataset=dataset)

    def __predict_available(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RektException("Fit must be executed before predict")
