import argparse
from dataclasses import dataclass

import pandas as pd

from fe.helpers import (
    add_centrality_window,
    add_dominant_frequencies,
    add_pca,
    add_signal_cutoff,
)
from utils.columns import Columns
from utils.parse import BaseArgs, get_base_parser

INPUT_PATH = "output/experiment_1/data/preprocessing.csv"
OUTPUT_PATH = "output/experiment_1/data/feature_engineering.csv"


@dataclass
class Args(BaseArgs):
    pass


def parse_args(args: list[str]) -> Args:
    base_parser = get_base_parser(INPUT_PATH, OUTPUT_PATH)
    parser = argparse.ArgumentParser(
        prog="fe",
        description="Feature engineering pipeline",
        parents=[base_parser],
    )

    arguments = parser.parse_args(args)
    return Args(arguments.input, arguments.output)


def run(args: Args) -> pd.DataFrame:
    print(f"Running feature engineering on {args.input}")

    df = pd.read_csv(args.input)
    # Make sure target doesn't get into feature selection
    target_key = Columns.get_target_column()
    target = df[target_key]
    df = df.drop(columns=target_key)

    # PCA
    n_comps = (3, 5, 12)
    for n_comp in n_comps:
        df = add_pca(
            df=df, feature_columns=Columns.get_feature_columns(), n_components=n_comp
        )

    # Rest
    for fe_fun in (add_centrality_window, add_dominant_frequencies, add_signal_cutoff):
        df = fe_fun(df=df, feature_columns=Columns.get_feature_columns())

    # Get target back in and drop rows with empty data (resulting from f.e.)
    df = pd.concat([df, target], axis="columns").dropna(axis=0, how="any")

    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    return df


def main(args: list[str]) -> None:
    run(parse_args(args))
