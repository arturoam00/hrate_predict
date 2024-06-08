import os

import pandas as pd


def get_exp_start_end(base_path: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Gets second argument of last line of time.csv (total duration)
    """
    path = os.path.join(base_path, "time.csv")
    assert os.path.exists(path), f"Unable to find {path}! Check downloaded files"
    times = pd.read_csv(path, parse_dates=["system time text"])[
        "system time text"
    ].to_numpy()
    return times[0], times[-1]
