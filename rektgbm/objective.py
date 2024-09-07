from dataclasses import dataclass

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType

OBJECTIVE_DICT_KEY: str = "objective"


class ObjectiveName(BaseEnum):
    rmse: str = "rmse"
    mae: str = "mae"
    huber: str = "huber"
    quantile: str = "quantile"
    gamma: str = "gamma"
    binary: str = "binary"
    multiclass: str = "multiclass"
    lambdarank: str = "lambdarank"
    ndcg: str = "ndcg"


class XgbObjectiveName(BaseEnum):
    squarederror: str = "reg:squarederror"
    squaredlogerror: str = "reg:squaredlogerror"
    pseudohubererror: str = "reg:pseudohubererror"
    absoluteerror: str = "reg:absoluteerror"
    quantileerror: str = "reg:quantileerror"
    logistic: str = "binary:logistic"
    logitraw: str = "binary:logitraw"
    hinge: str = "binary:hinge"
    poisson: str = "count:poisson"
    cox: str = "survival:cox"
    aft: str = "survival:aft"
    softmax: str = "multi:softmax"
    softprob: str = "multi:softprob"
    pairwise: str = "rank:pairwise"
    ndcg: str = "rank:ndcg"
    map: str = "rank:map"
    gamma: str = "reg:gamma"
    tweedie: str = "reg:tweedie"


class LgbObjectiveName(BaseEnum):
    rmse: str = "rmse"
    mae: str = "mae"
    huber: str = "huber"
    fair: str = "fair"
    poisson: str = "poisson"
    quantile: str = "quantile"
    mape: str = "mape"
    gamma: str = "gamma"
    tweedie: str = "tweedie"
    binary: str = "binary"
    multiclass: str = "multiclass"
    multiclassova: str = "multiclassova"
    cross_entropy: str = "cross_entropy"
    cross_entropy_lambda: str = "cross_entropy_lambda"
    lambdarank: str = "lambdarank"
    rank_xendcg: str = "rank_xendcg"


TASK_OBJECTIVE_MAPPER: dict[TaskType, list[ObjectiveName]] = {
    TaskType.regression: [
        ObjectiveName.rmse,
        ObjectiveName.mae,
        ObjectiveName.huber,
        ObjectiveName.quantile,
        ObjectiveName.gamma,
    ],
    TaskType.binary: [
        ObjectiveName.binary,
    ],
    TaskType.multiclass: [
        ObjectiveName.multiclass,
    ],
    TaskType.rank: [
        ObjectiveName.lambdarank,
        ObjectiveName.ndcg,
    ],
}


OBJECTIVE_ENGINE_MAPPER: dict[ObjectiveName, dict[MethodName, str]] = {
    ObjectiveName.rmse: {
        MethodName.lightgbm: LgbObjectiveName.rmse.value,
        MethodName.xgboost: XgbObjectiveName.squarederror.value,
    },
    ObjectiveName.mae: {
        MethodName.lightgbm: LgbObjectiveName.mae.value,
        MethodName.xgboost: XgbObjectiveName.absoluteerror.value,
    },
    ObjectiveName.huber: {
        MethodName.lightgbm: LgbObjectiveName.huber.value,
        MethodName.xgboost: XgbObjectiveName.pseudohubererror.value,
    },
    ObjectiveName.quantile: {
        MethodName.lightgbm: LgbObjectiveName.quantile.value,
        MethodName.xgboost: XgbObjectiveName.quantileerror.value,
    },
    ObjectiveName.gamma: {
        MethodName.lightgbm: LgbObjectiveName.gamma.value,
        MethodName.xgboost: XgbObjectiveName.gamma.value,
    },
    ObjectiveName.binary: {
        MethodName.lightgbm: LgbObjectiveName.binary.value,
        MethodName.xgboost: XgbObjectiveName.logistic.value,
    },
    ObjectiveName.multiclass: {
        MethodName.lightgbm: LgbObjectiveName.multiclass.value,
        MethodName.xgboost: XgbObjectiveName.softmax.value,
    },
    ObjectiveName.lambdarank: {
        MethodName.lightgbm: LgbObjectiveName.lambdarank.value,
        MethodName.xgboost: XgbObjectiveName.pairwise.value,
    },
    ObjectiveName.ndcg: {
        MethodName.lightgbm: LgbObjectiveName.rank_xendcg.value,
        MethodName.xgboost: XgbObjectiveName.ndcg.value,
    },
}


@dataclass
class RektObjective:
    task_type: TaskType
    objective: str | None

    def __post_init__(self) -> None:
        if self.objective:
            self.objective = ObjectiveName.get(self.objective)
            self.__validate_objective()
        else:
            _objectives = TASK_OBJECTIVE_MAPPER.get(self.task_type)
            self.objective = _objectives[0]

        self._objective_engine_mapper = OBJECTIVE_ENGINE_MAPPER.get(self.objective)

    def get_objective_str(self, method: MethodName) -> str:
        return self._objective_engine_mapper.get(method)

    def get_objective_dict(self, method: MethodName) -> dict[str, str]:
        return {OBJECTIVE_DICT_KEY: self.get_objective_str(method=method)}

    def __validate_objective(self) -> None:
        objectives = TASK_OBJECTIVE_MAPPER.get(self.task_type)
        if self.objective not in objectives:
            raise ValueError(
                f"Task type '{self.task_type}' and objective '{self.objective}' are not matched."
            )
