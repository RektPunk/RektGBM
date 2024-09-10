import numpy as np
import pandas as pd
import pytest

from rektgbm.encoder import RektLabelEncoder


# Test data for categorical variables
@pytest.fixture
def sample_data():
    return pd.Series(["apple", "banana", "orange", None, "kiwi", "Unseen", np.nan])


# Test data for label encoding
@pytest.fixture
def sample_label_data():
    return np.array([2, 3, 5, 0, 4, 1, 0])


def test_fit_transform(sample_data):
    encoder = RektLabelEncoder()
    transformed = encoder.fit_transform(sample_data)

    # Check that the transformed result is numeric
    assert transformed is not None
    assert transformed.dtype == int
    assert len(transformed) == len(sample_data)


def test_unseen_and_nan_values(sample_data):
    encoder = RektLabelEncoder()
    encoder.fit(sample_data)

    # Include new unseen value and check behavior
    test_data = pd.Series(["apple", "unknown", None, "melon", np.nan])
    transformed = encoder.transform(test_data)

    # Check for correct handling of unseen and NaN values
    assert (
        transformed
        == encoder.label_encoder.transform(["apple", "Unseen", "NaN", "Unseen", "NaN"])
    ).all()


def test_fit_transform_label(sample_label_data):
    encoder = RektLabelEncoder()
    transformed = encoder.fit_transform_label(sample_label_data)

    # Check if the labels are encoded correctly
    assert transformed is not None
    assert transformed.dtype == int
    assert np.array_equal(transformed, np.array([2, 3, 5, 0, 4, 1, 0]))


def test_inverse_transform_label(sample_label_data):
    encoder = RektLabelEncoder()
    encoder.fit_label(sample_label_data)
    transformed = encoder.transform_label(sample_label_data)
    inverse_transformed = encoder.inverse_transform(transformed)

    # Check if the inverse transform works correctly
    assert np.array_equal(inverse_transformed, sample_label_data)
