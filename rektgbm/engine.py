from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from rektgbm.base import BaseGBM, MethodName, RektException
from rektgbm.dataset import RektDataset
from rektgbm.metric import MetricName
from rektgbm.objective import ObjectiveName


class Engine(BaseGBM):
    def __init__(
        self,
        params: Dict[str, Any],
        objective: ObjectiveName,
        metric: MetricName,
        method: MethodName,
    ):
        self.params = params
        self.objective = objective
        self.metric = metric
        self.method = method

    def fit(
        self,
        dataset: RektDataset,
        valid_set: Optional[RektDataset] = None,
    ):
        params = self.params.update(
            {
                # TODO: lgb, xgb objective mapper, metric mapper
                "objective": self.objective,
                "metric": self.metric,  # can be set
            }
        )
        if valid_set is None:
            dtrain, dvalid = dataset.split(method=self.method)
        else:
            dtrain = dataset.dtrain(method=self.method)
            dvalid = valid_set.dtrain(method=self.method)

        if self.__is_lgb:
            self.model = lgb.train(
                train_set=dtrain,
                params=params,
                valid_sets=dvalid,
            )
        elif self.__is_xgb:
            self.model = xgb.train(
                dtrain=dtrain,
                verbose_eval=False,
                params=params,
                evals=[(dvalid, "valid")],
            )
        self._fitted = True

    def predict(
        self,
        dataset: RektDataset,
    ) -> np.ndarray:
        self.__predict_available()
        _pred = self.model.predict(data=dataset.dpredict(method=self.method))
        return _pred

    def __predict_available(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RektException("Fit must be executed before predict")

    @property
    def __is_lgb(self) -> bool:
        return self.method == MethodName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        return self.method == MethodName.xgboost
