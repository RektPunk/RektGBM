from dataclasses import dataclass
from typing import Dict, Optional, Union

import lightgbm as lgb
import xgboost as xgb

from rektgbm.base import (
    BaseEnum,
    DataFuncLike,
    DataLike,
    MethodName,
    XdataLike,
    YdataLike,
)


class TypeName(BaseEnum):
    train_dtype: int = 1
    predict_dtype: int = 2


MODEL_FUNC_TYPE_MAPPER: Dict[MethodName, Dict[TypeName, DataFuncLike]] = {
    MethodName.lightgbm: {
        TypeName.train_dtype: lgb.Dataset,
        TypeName.predict_dtype: lambda data: data,
    },
    MethodName.xgboost: {
        TypeName.train_dtype: xgb.DMatrix,
        TypeName.predict_dtype: xgb.DMatrix,
    },
}


@dataclass
class RektDataset:
    data: XdataLike
    label: Optional[YdataLike] = None
    model: str = MethodName.lightgbm.value

    def __post_init__(self):
        _model = MethodName.get(self.model)
        _funcs = MODEL_FUNC_TYPE_MAPPER.get(_model)
        self.train_dtype = _funcs.get(TypeName.train_dtype)
        self.predict_dtype = _funcs.get(TypeName.predict_dtype)
        self._is_none_label = True if self.label is None else False

    @property
    def dtrain(self) -> DataLike:
        self.__label_available()
        return self.train_dtype(data=self.data, label=self.label)

    @property
    def dpredict(self) -> Union[DataLike, XdataLike]:
        return self.predict_dtype(data=self.data)

    def __label_available(self) -> None:
        """Check if the label is available, raises an exception if not."""
        if getattr(self, "_is_none_label", False):
            raise AttributeError("Label is not available because it is set to None.")

    # TODO: validation split logic for optimize
