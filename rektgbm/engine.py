from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from rektgbm.base import BaseGBM, MethodName, StateException
from rektgbm.dataset import RektDataset
from rektgbm.metric import METRIC_DICT_KEY_MAPPER, LgbMetricName

_VALID_STR: str = "valid"


class RektEngine(BaseGBM):
    def __init__(
        self,
        method: MethodName,
        params: Dict[str, Any],
    ) -> None:
        self.method = method
        self.params = params

    def fit(
        self,
        dataset: RektDataset,
        valid_set: Optional[RektDataset],
    ) -> None:
        if valid_set is None:
            dtrain, dvalid = dataset.dsplit(method=self.method)
        else:
            dtrain = dataset.dtrain(method=self.method)
            dvalid = valid_set.dtrain(method=self.method)

        if self.__is_lgb:
            self.model = lgb.train(
                train_set=dtrain,
                params=self.params,
                valid_sets=dvalid,
                valid_names=_VALID_STR,
            )
        elif self.__is_xgb:
            evals_result = {}
            self.model = xgb.train(
                dtrain=dtrain,
                verbose_eval=False,
                params=self.params,
                evals_result=evals_result,
                evals=[(dvalid, _VALID_STR)],
            )
            self.evals_result = evals_result
        self._is_fitted = True

    def predict(
        self,
        dataset: RektDataset,
    ) -> np.ndarray:
        self.__check_fitted()
        _pred = self.model.predict(data=dataset.dpredict(method=self.method))
        return _pred

    @property
    def eval_loss(self) -> float:
        self.__check_fitted()
        metric_str = METRIC_DICT_KEY_MAPPER.get(self.method)
        if self.__is_lgb:
            _metric_name = self.params.get(metric_str)
            if _metric_name in {LgbMetricName.ndcg.value, LgbMetricName.map.value}:
                _metric_name = f"{_metric_name}@{self.params['eval_at']}"
            return float(self.model.best_score[_VALID_STR][_metric_name])
        elif self.__is_xgb:
            _metric_name = self.params.get(metric_str)
            return float(self.evals_result[_VALID_STR][_metric_name][-1])

    def __check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise StateException("Fit must be executed before predict")

    @property
    def __is_lgb(self) -> bool:
        return self.method == MethodName.lightgbm

    @property
    def __is_xgb(self) -> bool:
        return self.method == MethodName.xgboost
