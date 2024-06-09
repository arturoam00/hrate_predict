import os
import re
from typing import Callable, Union

import numpy as np
import pandas as pd
import pytz


class BaseLoader:
    """
    To load data from the raw .csv's

    The following must be implemented in children classes:
        - **FILENAME**: (class variable) (e.g. "Accelerometer.csv)
        - **column_function_map**: (property) (a map between column name and
            aggregation function for that column)

    This is basically to account for the variability in the column names and
    the potential different aggregation methods depending on 1) type of sensor
    and 2) type of variable measured by that sensor.

    Check the method `load()` to get an idea of what this is for
    """

    FILENAME = ""

    @property
    def column_function_map(self) -> dict[str, Callable]:
        pass

    @property
    def columns(self) -> list[str]:
        return list(self.column_function_map.keys())

    def __init__(self, *, base_path: str, date_range: pd.DatetimeIndex) -> None:
        self.path = self.check_path(base_path)
        self.date_range = date_range.sort_values()

    def check_path(self, base_path: str) -> str:
        path = os.path.join(base_path, self.FILENAME)
        assert os.path.exists(path), f"Provided file: {path} doesn't exist!"
        return path

    def parse_timekeys(self, df: pd.DataFrame) -> pd.DataFrame:
        time_key = "Time (s)"
        start = self.date_range[0]
        df[time_key] = pd.to_timedelta(df[time_key], unit="s") + start
        # Data from some sensors (heart rate) doesn't come with tz info
        df[time_key] = df[time_key].dt.tz_localize(None)
        return df.set_index(time_key)

    def aggregate(self, s: pd.Series, date_range: pd.DatetimeIndex) -> pd.Series:
        res = pd.Series(index=date_range)
        for t, t_next in zip(date_range, date_range[1:]):
            res[t] = self.column_function_map[s.name](s.loc[t:t_next])
        return res

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        base_name = self.FILENAME.rstrip(".csv") + "_"
        df.columns = [
            base_name
            + re.sub(pattern=r"\(.*\)", repl="", string=col).strip().replace(" ", "_")
            for col in df.columns
        ]
        return df

    def load(self) -> pd.DataFrame:
        res = pd.DataFrame()
        # Read raw file
        df = pd.read_csv(self.path)
        # Parse column with datetime information
        df = self.parse_timekeys(df)
        # Aggregate data from every column according to certain functiono
        for col in self.columns:
            res[col] = self.aggregate(df[col], self.date_range)
        res = self.rename_columns(res)
        return res

    def __str__(self) -> str:
        pass


class AccelerometerLoader(BaseLoader):

    FILENAME = "Accelerometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (m/s^2)": np.mean,
            "Y (m/s^2)": np.mean,
            "Z (m/s^2)": np.mean,
        }

    def __str__(self) -> str:
        return "Accelerometer Loader"


class BarometerLoader(BaseLoader):

    FILENAME = "Barometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {"X (hPa)": np.mean}

    def __str__(self) -> str:
        return "Barometer Loader"


class GyroscopeLoader(BaseLoader):

    FILENAME = "Gyroscope.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (rad/s)": np.mean,
            "Y (rad/s)": np.mean,
            "Z (rad/s)": np.mean,
        }

    def __str__(self) -> str:
        return "Gyroscope Loader"


class LinearAccelerometerLoader(BaseLoader):

    FILENAME = "Linear_Accelerometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (m/s^2)": np.mean,
            "Y (m/s^2)": np.mean,
            "Z (m/s^2)": np.mean,
        }

    def __str__(self) -> str:
        return "Linear Accelerometer Loader"


class LocationLoader(BaseLoader):

    FILENAME = "Location.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "Latitude (°)": self.get_last,
            "Longitude (°)": self.get_last,
            "Height (m)": self.get_last,
            "Velocity (m/s)": self.get_last,
            # Are this following any relevant ?
            # "Direction (°)": ...,
            # "Horizontal Accuracy (m)": ...,
            # "Vertical Accuracy (°)": ...,
        }

    def get_last(self, s: Union[pd.NA, pd.Series]):
        if not s.empty:
            return s.to_numpy()[-1]
        return np.nan

    def __str__(self) -> str:
        return "Location Loader"


class MagnetometerLoader(BaseLoader):

    FILENAME = "Magnetometer.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {
            "X (µT)": np.mean,
            "Y (µT)": np.mean,
            "Z (µT)": np.mean,
        }

    def __str__(self) -> str:
        return "Magnetometer Loader"


class ProximityLoader(BaseLoader):

    FILENAME = "Proximity.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {"Distance (cm)": np.mean}

    def __str__(self) -> str:
        return "Proximity Loader"


class HeartRateLoader(BaseLoader):

    FILENAME = "Heart_rate.csv"

    @property
    def column_function_map(self) -> dict[str, Callable]:
        return {"Avg (count/min)": np.mean}

    def parse_timekeys(self, df: pd.DataFrame) -> pd.DataFrame:
        time_key = "Date/Time"
        df[time_key] = pd.to_datetime(df[time_key])
        return df.set_index(time_key)


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
