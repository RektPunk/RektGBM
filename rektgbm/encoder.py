import numpy as np
from sklearn.preprocessing import LabelEncoder

from rektgbm.base import XdataLike, YdataLike


class RektLabelEncoder(object):
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()

    def fit(self, series: XdataLike) -> None:
        self.label_encoder.fit(list(series[~series.isna()]) + ["Unseen", "NaN"])

    def transform(self, series: XdataLike) -> XdataLike:
        return self.label_encoder.transform(
            np.select(
                [series.isna(), ~series.isin(self.label_encoder.classes_)],
                ["NaN", "Unseen"],
                series,
            )
        )

    def fit_transform(self, series: XdataLike) -> XdataLike:
        self.fit(series=series)
        return self.transform(series=series)

    def fit_label(self, series: YdataLike) -> YdataLike:
        self.label_encoder.fit(y=series)

    def transform_label(self, series: YdataLike) -> YdataLike:
        return self.label_encoder.transform(y=series)

    def fit_transform_label(self, series: YdataLike) -> YdataLike:
        self.fit_label(series=series)
        return self.transform_label(series=series)

    def inverse_transform(self, series: XdataLike) -> XdataLike:
        return self.label_encoder.inverse_transform(y=series)
