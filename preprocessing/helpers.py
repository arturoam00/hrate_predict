from typing import Generator

import pandas as pd

from preprocessing.loaders import (
    AccelerometerLoader,
    BarometerLoader,
    GyroscopeLoader,
    HeartRateLoader,
    LinearAccelerometerLoader,
    LocationLoader,
    MagnetometerLoader,
    ProximityLoader,
)
from preprocessing.time_parser import TimeParser


def load_all(
    base_data_path: str, time_parser: TimeParser, date_range: pd.DatetimeIndex
) -> pd.DataFrame:
    res = pd.DataFrame(index=date_range)

    def _load_all(
        base_data_path: str, time_parser: TimeParser, date_range: pd.DatetimeIndex
    ) -> Generator:
        for loader_class in (
            AccelerometerLoader,
            BarometerLoader,
            GyroscopeLoader,
            HeartRateLoader,
            LinearAccelerometerLoader,
            LocationLoader,
            MagnetometerLoader,
            ProximityLoader,
        ):
            loader = loader_class(base_data_path=base_data_path)
            yield loader.load(time_parser=time_parser, date_range=date_range)

    for df in _load_all(base_data_path, time_parser=time_parser, date_range=date_range):
        res = res.merge(df, right_index=True, left_index=True)
    return res.reset_index(names="Time")
