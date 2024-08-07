from dataclasses import dataclass
from typing import Dict, List, Optional

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType

OBJECTIVE_DICT_KEY: str = "objective"


class ObjectiveName(BaseEnum):
    # lgb
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

    # xgb
    squarederror: str = "reg:squarederror"
    squaredlogerror: str = "reg:squaredlogerror"
    pseudohubererror: str = "reg:pseudohubererror"
    absoluteerror: str = "reg:reg:absoluteerror"
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
    pairwise: str = "rank:pairwise"
    gamma: str = "reg:gamma"
    tweedie: str = "reg:tweedie"


class XgbObjectiveName(BaseEnum):
    # https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
    squarederror: str = "reg:squarederror"
    squaredlogerror: str = "reg:squaredlogerror"
    pseudohubererror: str = "reg:pseudohubererror"
    absoluteerror: str = "reg:reg:absoluteerror"
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
    pairwise: str = "rank:pairwise"
    gamma: str = "reg:gamma"
    tweedie: str = "reg:tweedie"


class LgbObjectiveName(BaseEnum):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters
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


TASK_OBJECTIVE_MAPPER: Dict[TaskType, List[ObjectiveName]] = {
    TaskType.regression: [
        ObjectiveName.rmse,
        ObjectiveName.mae,
        ObjectiveName.huber,
        ObjectiveName.fair,
        ObjectiveName.poisson,
        ObjectiveName.quantile,
        ObjectiveName.mape,
        ObjectiveName.gamma,
        ObjectiveName.tweedie,
        ObjectiveName.squarederror,
        ObjectiveName.squaredlogerror,
        ObjectiveName.pseudohubererror,
    ],
    TaskType.binary: [
        ObjectiveName.binary,
        ObjectiveName.logistic,
        ObjectiveName.logitraw,
        ObjectiveName.hinge,
        ObjectiveName.cross_entropy,
        ObjectiveName.cross_entropy_lambda,
    ],
    TaskType.multiclass: [
        ObjectiveName.multiclass,
        ObjectiveName.multiclassova,
        ObjectiveName.softmax,
        ObjectiveName.softprob,
    ],
    TaskType.rank: [
        ObjectiveName.lambdarank,
        ObjectiveName.rank_xendcg,
        ObjectiveName.pairwise,
        ObjectiveName.ndcg,
        ObjectiveName.map,
    ],
}


OBJECTIVE_ENGINE_MAPPER: Dict[ObjectiveName, Dict[MethodName, str]] = {
    ObjectiveName.rmse: {
        MethodName.lightgbm: LgbObjectiveName.rmse.value,
        MethodName.xgboost: XgbObjectiveName.squarederror.value,
    },
    ObjectiveName.mae: {
        MethodName.lightgbm: LgbObjectiveName.mae.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.huber: {
        MethodName.lightgbm: LgbObjectiveName.huber.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.fair: {
        MethodName.lightgbm: LgbObjectiveName.fair.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.poisson: {
        MethodName.lightgbm: LgbObjectiveName.poisson.value,
        MethodName.xgboost: XgbObjectiveName.poisson.value,
    },
    ObjectiveName.quantile: {
        MethodName.lightgbm: LgbObjectiveName.quantile.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.mape: {
        MethodName.lightgbm: LgbObjectiveName.mape.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.gamma: {
        MethodName.lightgbm: LgbObjectiveName.gamma.value,
        MethodName.xgboost: XgbObjectiveName.gamma.value,
    },
    ObjectiveName.tweedie: {
        MethodName.lightgbm: LgbObjectiveName.tweedie.value,
        MethodName.xgboost: XgbObjectiveName.tweedie.value,
    },
    ObjectiveName.binary: {
        MethodName.lightgbm: LgbObjectiveName.binary.value,
        MethodName.xgboost: XgbObjectiveName.logistic.value,
    },
    ObjectiveName.multiclass: {
        MethodName.lightgbm: LgbObjectiveName.multiclass.value,
        MethodName.xgboost: XgbObjectiveName.softmax.value,
    },
    ObjectiveName.multiclassova: {
        MethodName.lightgbm: LgbObjectiveName.multiclassova.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.cross_entropy: {
        MethodName.lightgbm: LgbObjectiveName.cross_entropy.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.cross_entropy_lambda: {
        MethodName.lightgbm: LgbObjectiveName.cross_entropy_lambda.value,
        MethodName.xgboost: None,  # XGBoost does not have a direct equivalent
    },
    ObjectiveName.lambdarank: {
        MethodName.lightgbm: LgbObjectiveName.lambdarank.value,
        MethodName.xgboost: XgbObjectiveName.pairwise.value,
    },
    ObjectiveName.rank_xendcg: {
        MethodName.lightgbm: LgbObjectiveName.rank_xendcg.value,
        MethodName.xgboost: XgbObjectiveName.ndcg.value,
    },
    ObjectiveName.squarederror: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.squarederror.value,
    },
    ObjectiveName.squaredlogerror: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.squaredlogerror.value,
    },
    ObjectiveName.pseudohubererror: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.pseudohubererror.value,
    },
    ObjectiveName.logistic: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.logistic.value,
    },
    ObjectiveName.logitraw: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.logitraw.value,
    },
    ObjectiveName.hinge: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.hinge.value,
    },
    ObjectiveName.cox: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.cox.value,
    },
    ObjectiveName.aft: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.aft.value,
    },
    ObjectiveName.softmax: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.softmax.value,
    },
    ObjectiveName.softprob: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.softprob.value,
    },
    ObjectiveName.pairwise: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.pairwise.value,
    },
    ObjectiveName.ndcg: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.ndcg.value,
    },
    ObjectiveName.map: {
        MethodName.lightgbm: None,  # LightGBM does not have a direct equivalent
        MethodName.xgboost: XgbObjectiveName.map.value,
    },
}


@dataclass
class RektObjective:
    task_type: TaskType
    objective: Optional[str]

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

    def get_objective(self, method: MethodName) -> Dict[str, str]:
        return {OBJECTIVE_DICT_KEY: self.get_objective_str(method=method)}

    def __validate_objective(self) -> None:
        objectives = TASK_OBJECTIVE_MAPPER.get(self.task_type)
        if self.objective not in objectives:
            raise ValueError(
                f"Task type '{self.task_type}' and objective '{self.objective}' are not matched."
            )
