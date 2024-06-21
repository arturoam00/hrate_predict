import pandas as pd

from utils.columns import Columns


def drop_experiment_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="all", subset=Columns.get_feature_columns())


def drop_missing_hrate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, subset=Columns.get_target_column())


def drop_prox_distance(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=Columns.PROXIMITY_DISTANCE)


def interpolate_rest(df: pd.DataFrame) -> pd.DataFrame:
    """
    This have been checked and it's sensible to do
    Just the columns 'Location_*' and 'Barometer_X' have
    some missing values here and there
    """
    return (
        df.set_index(Columns.get_datetime_column())
        .interpolate(method="time")
        .reset_index(names=Columns.get_datetime_column())
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for fun in (
        drop_experiment_time_gaps,
        drop_missing_hrate_rows,
        drop_prox_distance,
        interpolate_rest,
    ):
        df = fun(df)
    return df
