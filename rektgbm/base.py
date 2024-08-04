from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


class BaseEnum(Enum):
    @classmethod
    def get(cls, text: str) -> Enum:
        cls.__check_valid(text)
        return cls[text]

    @classmethod
    def __check_valid(cls, text: str) -> None:
        if text not in cls._member_names_:
            valid_members = ", ".join(cls._member_names_)
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


class ModelName(BaseEnum):
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"


XdataLike = Union[pd.DataFrame, pd.Series, np.ndarray]
YdataLike = Union[pd.Series, np.ndarray]
AlphaLike = Union[List[float], float]
ModelLike = Union[lgb.basic.Booster, xgb.Booster]
DtrainLike = Union[lgb.basic.Dataset, xgb.DMatrix]
DataFuncLike = Callable[[XdataLike, Optional[YdataLike]], Union[XdataLike, DtrainLike]]
