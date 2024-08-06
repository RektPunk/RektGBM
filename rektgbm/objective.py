from dataclasses import dataclass
from typing import Dict, List, Optional

from rektgbm.base import BaseEnum, MethodName
from rektgbm.task import TaskType

# TODO
# 1. add alias
# 2. TASK_OBJECTIVE_MAPPER: [0]: default
# 3. update objectives and metrics

## TODO
# create common objective mapper
# "rmse" -> "reg:squarederror", "rmse"
OBJECTIVE_DICT_KEY: str = "objective"


class ObjectiveName(BaseEnum):
    rmse: str = "rmse"
    binary: str = "binary"
    multiclass: str = "multiclass"


class XgbObjectiveName(BaseEnum):
    # https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
    squarederror: str = "reg:squarederror"  # regression with squared loss.
    squaredlogerror: str = "reg:squaredlogerror"  # regression with squared log loss.
    pseudohubererror: str = "reg:pseudohubererror"  # regression with Pseudo Huber loss.
    logistic: str = "binary:logistic"  # logistic regression for binary classification, output probability.
    logitraw: str = "binary:logitraw"  # logistic regression for binary classification, output score before logistic transformation.
    hinge: str = "binary:hinge"  # hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
    poisson: str = "count:poisson"  # Poisson regression for count data, output mean of Poisson distribution.
    cox: str = "survival:cox"  #: Cox proportional hazards model for survival analysis.
    aft: str = "survival:aft"  #: Accelerated Failure Time model for survival analysis.
    softmax: str = "multi:softmax"  #: set XGBoost to do multiclass classification using the softmax objective, output a class number.
    softprob: str = "multi:softprob"  #: same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
    pairwise: str = "rank:pairwise"  #: set XGBoost to do ranking task by minimizing the pairwise loss.
    ndcg: str = "rank:ndcg"  #: set XGBoost to do ranking task by minimizing the normalized discounted cumulative gain (NDCG) loss.
    map_: str = "rank:map"  #: set XGBoost to do ranking task by minimizing the mean average precision (MAP) loss.
    gamma: str = "reg:gamma"  #: gamma regression with log-link. Output is a mean of gamma distribution.
    tweedie: str = "reg:tweedie"  #: Tweedie regression with log-link. Output is a mean of Tweedie distribution.


class LgbObjectiveName(BaseEnum):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters
    regression: str = "regression"
    regression_l1: str = "regression_l1"
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
    TaskType.regression: [ObjectiveName.rmse],
    TaskType.binary: [ObjectiveName.binary],
    TaskType.multiclass: [ObjectiveName.multiclass],
}


OBJECTIVE_ENGINE_MAPPER: Dict[ObjectiveName, Dict[MethodName, str]] = {
    ObjectiveName.rmse: {
        MethodName.lightgbm: LgbObjectiveName.regression.value,
        MethodName.xgboost: XgbObjectiveName.squarederror.value,
    }
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
