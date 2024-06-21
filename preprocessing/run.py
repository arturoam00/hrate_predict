import argparse
import os
from dataclasses import dataclass

import pandas as pd

from preprocessing.clean import clean
from preprocessing.helpers import load_all
from preprocessing.time_parser import TimeParser
from utils.parse import BaseArgs, get_base_parser

INPUT_PATH = "data/experiment_1/"
OUTPUT_PATH = "output/experiment_1/data/preprocessing.csv"


@dataclass
class Args(BaseArgs):
    freq: str


def parse_freq(_freq: str) -> str:
    return f"{int(_freq)}ms"


def parse_args(args: list[str]) -> Args:
    base_parser = get_base_parser(INPUT_PATH, OUTPUT_PATH)
    parser = argparse.ArgumentParser(
        prog="preprocessing",
        description="Preprocessing raw files from the experiment.",
        parents=[base_parser],
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=parse_freq,
        help="Frequency to be used to aggregate the data in miliseconds (default %(default)s).",
        default="1000",
    )

    arguments = parser.parse_args(args)
    return Args(arguments.input, arguments.output, arguments.freq)


def run(args: Args) -> pd.DataFrame:
    print(f"Running preprocessing on {args.input} ...")

    # Create a time parser to handle certain rather annoying files
    time_parser = TimeParser(base_data_path=os.path.join(args.input, "meta"))
    start, end = time_parser.start, time_parser.end

    # Create date range with custom frequency with the start and end dates of the experiment
    date_range = pd.date_range(start=start, end=end, freq=args.freq)

    # Load, merge and clean the data from all sensors
    df = clean(
        load_all(
            base_data_path=args.input,
            time_parser=time_parser,
            date_range=date_range,
        )
    )

    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    return df


def main(args: list[str]) -> None:
    run(parse_args(args))
