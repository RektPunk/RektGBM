from typing import Callable

import optuna

from rektgbm.base import BaseEnum, MethodName, ParamsLike, StateException
from rektgbm.dataset import RektDataset
from rektgbm.engine import RektEngine
from rektgbm.metric import RektMetric
from rektgbm.objective import ObjectiveName, RektObjective
from rektgbm.param import METHOD_PARAMS_MAPPER, set_additional_params
from rektgbm.task import TaskType, check_task_type


class _RektMethods(BaseEnum):
    both: int = 1
    lightgbm: int = 2
    xgboost: int = 3


class RektOptimizer:
    def __init__(
        self,
        method: str = "both",
        task_type: str | None = None,
        objective: str | None = None,
        metric: str | None = None,
        params: list[Callable] | Callable | None = None,
        additional_params: ParamsLike = {},
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
        self.task_type = task_type
        self.metric = metric
        self.additional_params = additional_params

    def optimize_params(
        self,
        dataset: RektDataset,
        n_trials: int,
        valid_set: RektDataset | None = None,
    ) -> None:
        self._task_type: TaskType = check_task_type(
            target=dataset.label,
            group=dataset.group,
            task_type=self.task_type,
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
            _label_encoder = dataset.fit_transform_label()
            self._label_encoder_used = True

        self.num_class = (
            dataset.n_label
            if self.rekt_objective.objective == ObjectiveName.multiclass
            else None
        )
        if valid_set is None:
            if self._task_type == TaskType.rank:
                raise ValueError(
                    "A validation set must be provided when using the 'rank' task."
                )
            dataset, valid_set = dataset.split(task_type=self._task_type)
        else:
            if self.__is_label_encoder_used:
                valid_set.transform_label(label_encoder=_label_encoder)

        self.studies: dict[MethodName, optuna.Study] = {}
        for method, param in zip(self.method, self.params):
            _addtional_params = set_additional_params(
                objective=self.rekt_objective.objective,
                metric=self.rekt_metric.get_metric_str(method=method),
                method=method,
                params=self.additional_params,
                num_class=self.num_class,
            )
            _objective = self.rekt_objective.get_objective_dict(method=method)
            _metric = self.rekt_metric.get_metric_dict(method=method)

            def _study_func(trial: optuna.Trial) -> float:
                _param = param(trial=trial)
                _param.update({**_objective, **_metric, **_addtional_params})
                _engine = RektEngine(
                    params=_param,
                    method=method,
                    task_type=self._task_type,
                )
                _engine.fit(dataset=dataset, valid_set=valid_set)
                return _engine.eval_loss

            _direction = "maximize" if self._task_type == TaskType.rank else "minimize"
            study = optuna.create_study(
                study_name=f"Rekt_{method.value}",
                direction=_direction,
                load_if_exists=True,
            )
            study.optimize(_study_func, n_trials=n_trials)
            self.studies.update({method: study})
        self._is_optimized = True

    @property
    def best_params(self) -> dict[str, str | int | float | ParamsLike | None]:
        self.__check_optimized()
        best_method = min(self.studies, key=lambda k: self.studies[k].best_value)
        best_study = self.studies.get(best_method)
        _best_params = best_study.best_params
        _addtional_params = set_additional_params(
            params=self.additional_params,
            objective=self.rekt_objective.objective,
            method=best_method,
            metric=self.rekt_metric.get_metric_str(method=best_method),
            num_class=self.num_class,
        )
        _best_params.update({**_addtional_params})
        return {
            "method": best_method.value,
            "params": _best_params,
            "task_type": self._task_type.value,
            "objective": self.rekt_objective.objective.value,
            "metric": self.rekt_metric.metric.value,
        }

    def __check_optimized(self) -> None:
        """Check if the optimization process has been completed."""
        if not getattr(self, "_is_optimized", False):
            raise StateException("Optimization is not completed.")

    @property
    def __is_label_encoder_used(self):
        return getattr(self, "_label_encoder_used", False)
