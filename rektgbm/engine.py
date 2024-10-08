import lightgbm as lgb
import numpy as np
import xgboost as xgb

from rektgbm.base import BaseGBM, MethodName, ParamsLike, StateException
from rektgbm.dataset import RektDataset
from rektgbm.metric import METRIC_DICT_KEY_MAPPER, LgbMetricName
from rektgbm.task import TaskType

_VALID_STR: str = "valid"


class RektEngine(BaseGBM):
    def __init__(
        self,
        method: MethodName,
        params: ParamsLike,
        task_type: TaskType,
    ) -> None:
        self.method = method
        self.params = params
        self.task_type = task_type

    def fit(
        self,
        dataset: RektDataset,
        valid_set: RektDataset | None,
    ) -> None:
        if valid_set is None:
            dtrain, dvalid = dataset.dsplit(
                method=self.method, task_type=self.task_type
            )
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
        else:
            evals_result = {}
            self.model = xgb.train(
                dtrain=dtrain,
                num_boost_round=100,
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
        _metric_name = self.params.get(metric_str)
        if self.__is_lgb:
            if _metric_name in {LgbMetricName.ndcg.value, LgbMetricName.map.value}:
                _metric_name = f"{_metric_name}@{self.params['eval_at']}"
            return float(self.model.best_score[_VALID_STR][_metric_name])
        else:
            return float(self.evals_result[_VALID_STR][_metric_name][-1])

    def __check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise StateException("Fit must be executed before predict")

    @property
    def __is_lgb(self) -> bool:
        return self.method == MethodName.lightgbm
