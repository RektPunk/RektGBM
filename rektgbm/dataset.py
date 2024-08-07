from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from rektgbm.base import (
    BaseEnum,
    DataFuncLike,
    DataLike,
    MethodName,
    XdataLike,
    YdataLike,
)


class _TypeName(BaseEnum):
    train_dtype: int = 1
    predict_dtype: int = 2


_METHOD_FUNC_TYPE_MAPPER: Dict[MethodName, Dict[_TypeName, DataFuncLike]] = {
    MethodName.lightgbm: {
        _TypeName.train_dtype: lgb.Dataset,
        _TypeName.predict_dtype: lambda data: data,
    },
    MethodName.xgboost: {
        _TypeName.train_dtype: xgb.DMatrix,
        _TypeName.predict_dtype: xgb.DMatrix,
    },
}


def _get_dtype(method: MethodName, dtype: _TypeName):
    _funcs = _METHOD_FUNC_TYPE_MAPPER.get(method)
    return _funcs.get(dtype)


def _train_valid_split(
    data: XdataLike, label: YdataLike
) -> Tuple[XdataLike, XdataLike, YdataLike, YdataLike]:
    return train_test_split(data, label, test_size=0.2, random_state=42)


@dataclass
class RektDataset:
    data: XdataLike
    label: Optional[YdataLike] = None

    def __post_init__(self):
        self.label = pd.Series(self.label)
        self._is_none_label = True if self.label is None else False

    def dtrain(self, method: MethodName) -> DataLike:
        self.__check_label_available()
        train_dtype = _get_dtype(
            method=method,
            dtype=_TypeName.train_dtype,
        )
        return train_dtype(data=self.data, label=self.label)

    def dpredict(self, method: MethodName) -> Union[DataLike, XdataLike]:
        predict_dtype = _get_dtype(
            method=method,
            dtype=_TypeName.predict_dtype,
        )
        return predict_dtype(data=self.data)

    def split(self) -> Tuple["RektDataset", "RektDataset"]:
        self.__check_label_available()
        train_data, valid_data, train_label, valid_label = _train_valid_split(
            data=self.data, label=self.label
        )
        dtrain = RektDataset(data=train_data, label=train_label)
        dvalid = RektDataset(data=valid_data, label=valid_label)
        return dtrain, dvalid

    def dsplit(self, method: MethodName) -> Tuple[DataLike, DataLike]:
        self.__check_label_available()
        train_data, valid_data, train_label, valid_label = _train_valid_split(
            data=self.data, label=self.label
        )
        train_dtype = _get_dtype(
            method=method,
            dtype=_TypeName.train_dtype,
        )
        dtrain = train_dtype(data=train_data, label=train_label)
        dvalid = train_dtype(data=valid_data, label=valid_label)
        return dtrain, dvalid

    @property
    def n_label(self) -> int:
        return int(self.label.nunique())

    def __check_label_available(self) -> None:
        """Check if the label is available, raises an exception if not."""
        if getattr(self, "_is_none_label", False):
            raise AttributeError("Label is not available because it is set to None.")
