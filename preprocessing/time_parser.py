import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeTuple:
    exp: float | pd.Timedelta
    real: str | pd.Timestamp

    def __post_init__(self):
        self.exp = pd.to_timedelta(self.exp, unit="s")
        self.real = pd.to_datetime(self.real).tz_localize(None)


@dataclass
class Interval:
    start: TimeTuple
    end: TimeTuple


class TimeParser:

    FILENAME = "time.csv"

    @property
    def columns(self) -> list[str]:
        return ["event", "experiment time", "system time text"]

    def __init__(self, base_data_path: str) -> None:
        self.path = os.path.join(base_data_path, self.FILENAME)
        self.intervals = self.extract_intervals()
        self.start = min(self.intervals, key=lambda x: x.start.real).start.real
        self.end = max(self.intervals, key=lambda x: x.end.real).end.real

    def extract_intervals(self) -> tuple[Interval]:
        res = tuple()
        df = pd.read_csv(self.path)
        subset_cols = ["experiment time", "system time text"]
        # start happens every two rows, same for pause
        start, pause = df[::2], df[1::2]
        for (start_exp, start_real), (pause_exp, pause_real) in zip(
            start[subset_cols].values, pause[subset_cols].values
        ):
            res += (
                Interval(
                    start=TimeTuple(exp=start_exp, real=start_real),
                    end=TimeTuple(exp=pause_exp, real=pause_real),
                ),
            )

        return sorted(res, key=lambda x: x.start.real)

    def parse_times(self, timekeys: pd.Series) -> pd.Series:
        timekeys = pd.to_timedelta(timekeys, unit="s")
        joined_intervals = []
        for interval in self.intervals:
            joined_intervals.append(
                # Take just the slice of the original timekeys that lie inside the interval
                timekeys[
                    (timekeys >= interval.start.exp) & (timekeys < interval.end.exp)
                ]
                # Take into account the start time of the interval as an offset
                + (interval.start.real - interval.start.exp)
            )

        return pd.concat(joined_intervals)

    def __call__(self, timekeys: pd.Series) -> pd.Series:
        return self.parse_times(timekeys=timekeys)
