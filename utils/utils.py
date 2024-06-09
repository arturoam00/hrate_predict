import os
from typing import Generator

import pandas as pd

from utils.loaders import ALL_LOADERS


def get_exp_start_end(base_path: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Gets first and last datetimes of the column `system time text` in time.csv
    this is, start and end of experiment
    """
    path = os.path.join(base_path, "time.csv")
    assert os.path.exists(path), f"Unable to find {path}! Check downloaded files"
    times = (
        pd.read_csv(path, parse_dates=["system time text"])["system time text"]
        .dt.tz_localize(
            None
        )  # Data from some sensors (heart rate) doesn't come wiht tz info
        .to_numpy()
    )
    return times[0], times[-1]


def load_all(base_path: str, date_range: pd.DatetimeIndex) -> Generator:
    """
    Returns generator with processed data frames for each of the sensor loaders
    """
    for loader_class in ALL_LOADERS:
        loader = loader_class(base_path=base_path, date_range=date_range)
        yield loader.load()


def concatenate_all(base_path: str, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    res = pd.DataFrame(index=date_range)
    for df in load_all(base_path, date_range):
        res = res.merge(df, right_index=True, left_index=True)
    return res.reset_index(names="Time")
