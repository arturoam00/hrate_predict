import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import pandas as pd

from preprocessing.time_parser import TimeParser

__all__ = (
    "AccelerometerLoader",
    "BarometerLoader",
    "GyroscopeLoader",
    "LinearAccelerometerLoader",
    "LocationLoader",
    "MagnetometerLoader",
    "ProximityLoader",
    "HeartRateLoader",
)


def median(s: pd.Series) -> Union[pd.NA, float]:
    if not s.empty:
        return s.median()
    return pd.NA


def get_last(s: Union[pd.NA, pd.Series]):
    if not s.empty:
        return s.to_numpy()[-1]
    return np.nan


class BaseLoader(ABC):

    FILENAME = ""

    @property
    @abstractmethod
    def column_function_map(self) -> dict[str, Callable]:
        pass

    @property
    def columns(self) -> list[str]:
        return list(self.column_function_map.keys())

    @property
    def POST_LOAD_FUNS(self) -> tuple[Callable]:
        """
        Tuple with functions to apply to the loaded data frame
        Order matters, of course first function is applied first
        """
        return (self.rename_columns,)

    def __init__(self, *, base_data_path: str) -> None:
        self.path = os.path.join(base_data_path, self.FILENAME)
        self.time_parser = None
        self.date_range = None

    @abstractmethod
    def parse_timekeys(self, *args, **kwargs) -> pd.DataFrame:
        pass

    def aggregate(
        self, *, df: pd.DataFrame, date_range: pd.DatetimeIndex
    ) -> pd.DataFrame:
        res = pd.DataFrame()

        def _aggregate(s: pd.Series, date_range: pd.DatetimeIndex) -> pd.Series:
            res = pd.Series(index=date_range)
            for t, t_next in zip(date_range, date_range[1:]):
                res[t] = self.column_function_map[s.name](s.loc[t:t_next])
            return res

        for col in self.columns:
            res[col] = _aggregate(df[col], date_range=date_range)
        return res

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        base_name = self.FILENAME.rstrip(".csv").replace(" ", "_") + "_"
        df.columns = [
            base_name
            + re.sub(pattern=r"\(.*\)", repl="", string=col).strip().replace(" ", "_")
            for col in df.columns
        ]
        return df

    def load(
        self,
        *,
        time_parser: TimeParser | None = None,
        date_range: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        self.time_parser = time_parser
        self.date_range = date_range
        # Load and apply post loading functions
        df = self._load()
        for fun in self.POST_LOAD_FUNS:
            df = fun(df)
        return df

    def _load(
        self,
    ) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df = self.parse_timekeys(
            df=pd.read_csv(self.path), time_parser=self.time_parser
        )
        # If date_range provided, assume we want to aggregate
        if self.date_range is not None:
            df = self.aggregate(df=df, date_range=self.date_range)
        return df

    def __str__(self) -> str:
        pass


class BasePhyphoxLoader(BaseLoader):

    def parse_timekeys(
        self, *, df: pd.DataFrame, time_parser: TimeParser
    ) -> pd.DataFrame:
        time_key = "Time (s)"
        df[time_key] = time_parser(timekeys=df[time_key])
        df[time_key] = df[time_key].dt.tz_localize(None)
        df = df.set_index(time_key)
        return df[df.index.notnull()]


class BaseAppleWatchLoader(BaseLoader):

    def parse_timekeys(self, *, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        time_key = "Date/Time"
        df[time_key] = pd.to_datetime(df[time_key])
        return df.set_index(time_key)


class AccelerometerLoader(BasePhyphoxLoader):

    FILENAME = "Accelerometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (m/s^2)": median,
            "Y (m/s^2)": median,
            "Z (m/s^2)": median,
        }

    def __str__(self) -> str:
        return "Accelerometer Loader"


class BarometerLoader(BasePhyphoxLoader):

    FILENAME = "Barometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {"X (hPa)": median}

    def __str__(self) -> str:
        return "Barometer Loader"


class GyroscopeLoader(BasePhyphoxLoader):

    FILENAME = "Gyroscope.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (rad/s)": median,
            "Y (rad/s)": median,
            "Z (rad/s)": median,
        }

    def __str__(self) -> str:
        return "Gyroscope Loader"


class LinearAccelerometerLoader(BasePhyphoxLoader):

    FILENAME = "Linear Accelerometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (m/s^2)": median,
            "Y (m/s^2)": median,
            "Z (m/s^2)": median,
        }

    def __str__(self) -> str:
        return "Linear Accelerometer Loader"


class LocationLoader(BasePhyphoxLoader):

    FILENAME = "Location.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "Latitude (°)": get_last,
            "Longitude (°)": get_last,
            "Height (m)": get_last,
            "Velocity (m/s)": get_last,
            # Are this following any relevant ?
            # "Direction (°)": ...,
            # "Horizontal Accuracy (m)": ...,
            # "Vertical Accuracy (°)": ...,
        }

    def __str__(self) -> str:
        return "Location Loader"


class MagnetometerLoader(BasePhyphoxLoader):

    FILENAME = "Magnetometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (µT)": median,
            "Y (µT)": median,
            "Z (µT)": median,
        }

    def __str__(self) -> str:
        return "Magnetometer Loader"


class ProximityLoader(BasePhyphoxLoader):

    FILENAME = "Proximity.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {"Distance (cm)": median}

    def __str__(self) -> str:
        return "Proximity Loader"


class HeartRateLoader(BaseAppleWatchLoader):

    FILENAME = "Heart_rate.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {"Avg (count/min)": median}

    @property
    def POST_LOAD_FUNS(self) -> tuple[Callable]:
        return (self.fillna_hrate,) + super().POST_LOAD_FUNS

    def fillna_hrate(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Avg (count/min)"] = df["Avg (count/min)"].interpolate(method="time")
        return df

    def __str__(self) -> str:
        return "Heart Rate Loader"
