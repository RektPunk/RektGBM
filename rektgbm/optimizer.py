from typing import Any, Callable, Dict, List, Optional, Union

import optuna

from rektgbm.addparams import set_additional_params
from rektgbm.base import BaseEnum, MethodName, StateException
from rektgbm.dataset import RektDataset
from rektgbm.engine import RektEngine
from rektgbm.metric import RektMetric
from rektgbm.objective import RektObjective
from rektgbm.param import METHOD_PARAMS_MAPPER
from rektgbm.task import check_task_type


class _RektMethods(BaseEnum):
    both: int = 1
    lightgbm: int = 2
    xgboost: int = 3


class RektOptimizer:
    def __init__(
        self,
        method: str = "both",
        task_type: Optional[str] = None,
        objective: Optional[str] = None,
        metric: Optional[str] = None,
        params: Optional[Union[List[Callable], Callable]] = None,
        additional_params: Dict[str, Any] = {},
    ) -> None:
        if _RektMethods.both == _RektMethods.get(method):
            self.method = [MethodName.lightgbm, MethodName.xgboost]
        else:
            self.method = [MethodName.get(method)]

        if params:
            if isinstance(params, Callable):
                params = [params]
            self.params = params
            assert len(self.method) == len(
                self.params
            ), "Length of methods are not same that of params"
        else:
            self.params = [METHOD_PARAMS_MAPPER.get(method) for method in self.method]
        self.objective = objective
        self._task_type = task_type
        self.metric = metric
        self.additional_params = additional_params

    def optimize_params(
        self,
        dataset: RektDataset,
        n_trials: int,
        valid_set: Optional[RektDataset] = None,
    ) -> Dict[str, Any]:
        self.task_type = check_task_type(
            target=dataset.label,
            task_type=self._task_type,
        )
        self.rekt_objective = RektObjective(
            task_type=self.task_type,
            objective=self.objective,
        )
        self.rekt_metric = RektMetric(
            task_type=self.task_type,
            objective=self.rekt_objective.objective,
            metric=self.metric,
        )
        self.studies: Dict[MethodName, optuna.Study] = {}
        for method, param in zip(self.method, self.params):
            if valid_set is None:
                dataset, valid_set = dataset.split()

            def _study_func(trial: optuna.Trial) -> float:
                _param = param(trial=trial)
                _objective = self.rekt_objective.get_objective(method=method)
                _metric = self.rekt_metric.get_metric(method=method)
                _addtional_params = set_additional_params(
                    objective=self.rekt_objective.objective,
                    method=method,
                    params=self.additional_params,
                )
                _param.update({**_objective, **_metric, **_addtional_params})

                _engine = RektEngine(
                    params=_param,
                    method=method,
                )
                _engine.fit(dataset=dataset, valid_set=valid_set)
                return _engine.eval_loss

            study = optuna.create_study(
                study_name=f"Rekt_{method.value}",
                direction="minimize",
                load_if_exists=True,
            )
            study.optimize(_study_func, n_trials=n_trials)
            self.studies.update({method: study})
        self._is_optimized = True

    @property
    def best_params(self) -> Dict[str, Any]:
        self.__check_optimized()
        best_method = min(self.studies, key=lambda k: self.studies[k].best_value)
        best_study = self.studies.get(best_method)
        _best_params = best_study.best_params
        _addtional_params = set_additional_params(
            objective=self.rekt_objective.objective,
            method=best_method,
            params=self.additional_params,
        )
        _best_params.update({**_addtional_params})
        return {
            "method": best_method.value,
            "params": _best_params,
            "task_type": self.task_type.value,
            "objective": self.rekt_objective.objective.value,
            "metric": self.rekt_metric.metric.value,
        }

    def __check_optimized(self) -> None:
        """Check if the optimization process has been completed."""
        if not getattr(self, "_is_optimized", False):
            raise StateException("Optimization is not completed.")
