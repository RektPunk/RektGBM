from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import lightgbm as lgb
import xgboost as xgb

from rektgbm.base import (
    BaseEnum,
    DataFuncLike,
    DtrainLike,
    ModelName,
    XdataLike,
    YdataLike,
)


class TypeName(BaseEnum):
    train_dtype: str = "train_dtype"
    predict_dtype: str = "predict_dtype"


def _lgb_predict_dtype(data: XdataLike):
    return data


MODEL_FUNC_TYPE_MAPPER: Dict[ModelName, Dict[TypeName, DataFuncLike]] = {
    ModelName.lightgbm: {
        TypeName.train_dtype: lgb.Dataset,
        TypeName.predict_dtype: _lgb_predict_dtype,
    },
    ModelName.xgboost: {
        TypeName.train_dtype: xgb.DMatrix,
        TypeName.predict_dtype: xgb.DMatrix,
    },
}


@dataclass
class RektDataset:
    data: XdataLike
    label: Optional[YdataLike] = (None,)
    model: str = (ModelName.lightgbm.value,)

    def __post_init__(self):
        _model = ModelName.get(self.model)
        _funcs = MODEL_FUNC_TYPE_MAPPER.get(_model)
        self._train_dtype = _funcs.get(TypeName.train_dtype)
        self._predict_dtype = _funcs.get(TypeName.predict_dtype)
        self._is_none_label = True if self.label is None else False

    @property
    def train_dtype(self) -> Callable:
        return self._train_dtype

    @property
    def predict_dtype(self) -> Callable:
        return self._predict_dtype

    @property
    def dtrain(self) -> DtrainLike:
        self.__label_available()
        return self._train_dtype(data=self._data, label=self._label)

    @property
    def dpredict(self) -> Union[DtrainLike, Callable]:
        return self._predict_dtype(data=self._data)

    def __label_available(self) -> None:
        """Check if the label is available, raises an exception if not."""
        if getattr(self, "_is_none_label", False):
            raise AttributeError("Label is not available because it is set to None.")
