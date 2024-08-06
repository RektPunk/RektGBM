from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


class BaseEnum(Enum):
    @classmethod
    def get(cls, text: str) -> "BaseEnum":
        cls.__check_valid(text)
        return cls[text]

    @classmethod
    def __check_valid(cls, text: str) -> None:
        if text not in cls._member_map_.keys():
            valid_members = ", ".join(list(cls._member_map_.keys()))
            raise ValueError(
                f"Invalid value: '{text}'. Expected one of: {valid_members}."
            )


class BaseGBM(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class MethodName(BaseEnum):
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"
    lgb: str = "lightgbm"
    xgb: str = "xgboost"


XdataLike = Union[pd.DataFrame, pd.Series, np.ndarray]
YdataLike = Union[pd.Series, np.ndarray]
ModelLike = Union[lgb.basic.Booster, xgb.Booster]
DataLike = Union[lgb.basic.Dataset, xgb.DMatrix]
DataFuncLike = Callable[[XdataLike, Optional[YdataLike]], Union[DataLike, XdataLike]]


class StateException(Exception):
    pass
