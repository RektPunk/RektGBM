import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

from rektgbm import RektDataset
from rektgbm.base import MethodName
from rektgbm.encoder import RektLabelEncoder
from rektgbm.task import TaskType

# Example data
regression_data, regression_label = make_regression(
    n_samples=1_000, n_features=5, n_informative=3
)
classificatation_data, classification_label = make_classification(
    n_samples=1_000, n_features=5, n_informative=3, n_classes=3
)


# Test RektDataset __post_init__
def test_rektdataset_post_init():
    dataset = RektDataset(data=classificatation_data, label=classification_label)
    assert isinstance(dataset.data, pd.DataFrame)
    assert all(
        isinstance(encoder, RektLabelEncoder) for encoder in dataset.encoders.values()
    )

    dataset = RektDataset(
        data=regression_data, label=regression_label, skip_post_init=True
    )
    assert isinstance(dataset, np.ndarray)


# Test fit_transform_label
def test_rektdataset_fit_transform_label():
    dataset = RektDataset(data=classificatation_data, label=classification_label)
    label_encoder = dataset.fit_transform_label()
    assert isinstance(label_encoder, RektLabelEncoder)
    assert dataset._is_transformed is True


# Test transform_label
def test_rektdataset_transform_label():
    dataset = RektDataset(data=classificatation_data, label=classification_label)
    label_encoder = RektLabelEncoder()
    label_encoder.fit_label(classification_label)
    dataset.transform_label(label_encoder)
    assert (dataset.label == label_encoder.transform_label(classification_label)).all()


# Test dtrain method for lightgbm and xgboost
@pytest.mark.parametrize(
    "method, expected_type",
    [
        (MethodName.lightgbm, lgb.basic.Dataset),
        (MethodName.xgboost, xgb.DMatrix),
    ],
)
def test_rektdataset_dtrain(method, expected_type):
    dataset = RektDataset(data=classificatation_data, label=regression_label)
    dtrain = dataset.dtrain(method)
    assert isinstance(dtrain, expected_type)


# Test dpredict method for lightgbm and xgboost
@pytest.mark.parametrize(
    "method, expected_type",
    [
        (MethodName.lightgbm, pd.DataFrame),
        (MethodName.xgboost, xgb.DMatrix),
    ],
)
def test_rektdataset_dpredict(method, expected_type):
    dataset = RektDataset(data=regression_data, label=regression_label)
    dpredict = dataset.dpredict(method)
    assert isinstance(dpredict, expected_type)


# Test split method
def test_rektdataset_split():
    dataset = RektDataset(data=regression_data, label=regression_label)
    dtrain, dvalid = dataset.split(task_type=TaskType.regression)
    assert isinstance(dtrain, RektDataset)
    assert isinstance(dvalid, RektDataset)


# Test dsplit method for lightgbm and xgboost
@pytest.mark.parametrize(
    "method, expected_type",
    [
        (MethodName.lightgbm, lgb.basic.Dataset),
        (MethodName.xgboost, xgb.DMatrix),
    ],
)
def test_rektdataset_dsplit(method, expected_type):
    dataset = RektDataset(data=classificatation_data, label=classification_label)
    dtrain, dvalid = dataset.dsplit(method=method, task_type=TaskType.multiclass)
    assert isinstance(dtrain, expected_type)
    assert isinstance(dvalid, expected_type)


# Test n_label property
def test_rektdataset_n_label():
    dataset = RektDataset(data=classificatation_data, label=classification_label)
    assert dataset.n_label == 3


# Test __check_label_available
def test_rektdataset_check_label_available():
    dataset = RektDataset(data=classificatation_data)
    with pytest.raises(AttributeError):
        dataset.__check_label_available()
