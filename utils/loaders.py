import os
from typing import Callable

import numpy as np
import pandas as pd


class BaseLoader:

    FILENAME = ""

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def column_function_map(self) -> list[str, Callable]:
        pass

    def __init__(self, *, base_path: str, date_range: pd.DatetimeIndex) -> None:
        self.path = self.check_path(base_path)
        self.date_range = date_range.sort_values()

    @classmethod
    def check_path(cls, base_path: str) -> str:
        path = os.path.join(base_path, cls.FILENAME)
        assert os.path.exists(path), f"Provided file: {path} doesn't exist!"
        return path

    def parse_timedeltas(self, df: pd.DataFrame) -> pd.DataFrame:
        time_key = "Time (s)"
        start = self.date_range[0]
        df[time_key] = pd.to_timedelta(df[time_key], unit="s") + start
        return df.set_index(time_key)

    def aggregate(self, s: pd.Series, date_range: pd.DatetimeIndex) -> pd.Series:
        res = pd.Series(index=date_range)
        for t, t_next in zip(date_range, date_range[1:]):
            res[t] = self.column_function_map[s.name](s.loc[t:t_next])
        return res

    def load(self) -> pd.DataFrame:
        res = pd.DataFrame()
        df = pd.read_csv(self.path)
        df = self.parse_timedeltas(df)
        for col in self.columns:
            res[col] = self.aggregate(df[col], self.date_range)
        return res


class AccelerometerLoader(BaseLoader):

    FILENAME = "Accelerometer.csv"

    @property
    def columns(self) -> list[str]:
        return ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (m/s^2)": np.mean,
            "Y (m/s^2)": np.mean,
            "Z (m/s^2)": np.mean,
        }


class BarometerLoader(BaseLoader):

    FILENAME = ""

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def column_function_map(self) -> dict[str, Callable]:
        pass


class GyroscopeLoader(BaseLoader):

    FILENAME = ""

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def column_function_map(self) -> dict[str, Callable]:
        pass


class LinearAccelerometerLoader(BaseLoader):

    FILENAME = ""

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def column_function_map(self) -> dict[str, Callable]:
        pass


class LocationLoader(BaseLoader):

    FILENAME = ""

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def column_function_map(self) -> dict[str, Callable]:
        pass


class MagnetometerLoader(BaseLoader):

    FILENAME = ""

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def column_function_map(self) -> dict[str, Callable]:
        pass


class ProximityLoader(BaseLoader):

    FILENAME = ""

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def column_function_map(self) -> dict[str, Callable]:
        pass
