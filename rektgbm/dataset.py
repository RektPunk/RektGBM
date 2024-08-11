from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
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
from rektgbm.encoder import RektLabelEncoder
from rektgbm.task import TaskType


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
    data: XdataLike, label: YdataLike, task_type: TaskType
) -> Tuple[XdataLike, XdataLike, YdataLike, YdataLike]:
    if task_type == TaskType.regression:
        for _bin in range(5, 0, -1):
            try:
                _stratify = pd.cut(label, bins=_bin, labels=False)
            except:
                continue
    else:
        _stratify = label
    return train_test_split(
        data, label, test_size=0.2, random_state=42, stratify=_stratify
    )


@dataclass
class RektDataset:
    data: XdataLike
    label: Optional[YdataLike] = None
    group: Optional[YdataLike] = None
    reference: Optional["RektDataset"] = None
    skip_post_init: bool = False

    def __post_init__(self) -> None:
        self._is_none_label = True if self.label is None else False
        if self.skip_post_init:
            return

        if not isinstance(self.data, pd.DataFrame):
            self.data = pd.DataFrame(self.data)

        if self.reference is None:
            self.encoders: Dict[str, RektLabelEncoder] = {}
            for col in self.data.columns:
                if self.data[col].dtype == "object":
                    _encoder = RektLabelEncoder()
                    self.data[col] = _encoder.fit_transform(self.data[col])
                    self.encoders.update({col: _encoder})
        else:
            for col, _encoder in self.reference.encoders.items():
                self.data[col] = _encoder.transform(self.data[col])

    def fit_transform_label(self) -> RektLabelEncoder:
        if self.__is_label_transformed:
            return self.label_encoder
        self.label_encoder = RektLabelEncoder()
        self.label = self.label_encoder.fit_transform_label(series=self.label)
        self._is_transformed = True
        return self.label_encoder

    def transform_label(self, label_encoder: RektLabelEncoder) -> None:
        self.label = label_encoder.transform_label(series=self.label)

    def dtrain(self, method: MethodName) -> DataLike:
        self.__check_label_available()
        train_dtype = _get_dtype(
            method=method,
            dtype=_TypeName.train_dtype,
        )
        return train_dtype(data=self.data, label=self.label, group=self.group)

    def dpredict(self, method: MethodName) -> Union[DataLike, XdataLike]:
        predict_dtype = _get_dtype(
            method=method,
            dtype=_TypeName.predict_dtype,
        )
        return predict_dtype(data=self.data)

    def split(self, task_type: TaskType) -> Tuple["RektDataset", "RektDataset"]:
        self.__check_label_available()
        train_data, valid_data, train_label, valid_label = _train_valid_split(
            data=self.data,
            label=self.label,
            task_type=task_type,
        )
        dtrain = RektDataset(data=train_data, label=train_label, skip_post_init=True)
        dvalid = RektDataset(data=valid_data, label=valid_label, skip_post_init=True)
        return dtrain, dvalid

    def dsplit(
        self, method: MethodName, task_type: TaskType
    ) -> Tuple[DataLike, DataLike]:
        self.__check_label_available()
        train_data, valid_data, train_label, valid_label = _train_valid_split(
            data=self.data,
            label=self.label,
            task_type=task_type,
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
        if isinstance(self.label, pd.Series):
            return int(self.label.nunique())
        elif isinstance(self.label, np.ndarray):
            return len(np.unique(self.label))

    def __check_label_available(self) -> None:
        """Check if the label is available, raises an exception if not."""
        if getattr(self, "_is_none_label", False):
            raise AttributeError("Label is not available because it is set to None.")

    @property
    def __is_label_transformed(self) -> bool:
        return getattr(self, "_is_transformed", False)
