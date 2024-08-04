from typing import Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from rektgbm.base import DataLike, MethodName, XdataLike, YdataLike


def test_base_enum_get_valid():
    assert MethodName.get("lightgbm") == MethodName.lightgbm
    assert MethodName.get("xgboost") == MethodName.xgboost
    with pytest.raises(ValueError):
        MethodName.get("invalid")


def test_base_enum_check_valid():
    MethodName._BaseEnum__check_valid("lightgbm")
    MethodName._BaseEnum__check_valid("xgboost")
    with pytest.raises(ValueError):
        MethodName._BaseEnum__check_valid("invalid")


def test_xdatalike():
    assert isinstance(pd.DataFrame(), XdataLike)
    assert isinstance(pd.Series(), XdataLike)
    assert isinstance(np.ndarray([]), XdataLike)


def test_ydatalike():
    assert isinstance(pd.Series(), YdataLike)
    assert isinstance(np.ndarray([]), YdataLike)


def test_dtrainlike():
    _dummy_data = np.array([[1, 2], [3, 4]])
    _dummy_label = np.array([0, 1])
    dummy_lgb_dset = lgb.Dataset(data=_dummy_data, label=_dummy_label)
    dummy_xgb_dmat = xgb.DMatrix(data=_dummy_data, label=_dummy_label)
    assert isinstance(dummy_lgb_dset, DataLike)
    assert isinstance(dummy_xgb_dmat, DataLike)
