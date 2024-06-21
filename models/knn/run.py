import argparse
import json
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from models.base import RegressionModelRunner
from utils.parse import (
    BaseArgs,
    GridArgs,
    get_base_parser,
    get_grid_and_single_subparsers,
)

INPUT_PATH = "output/experiment_1/data/feature_engineering.csv"
OUTPUT_PATH = "output/experiment_1/models/knn.txt"


@dataclass
class KnnArgs(BaseArgs):
    neighbors: int
    weights: str
    metric: str


def parse_single(args: argparse.Namespace) -> KnnArgs:
    return KnnArgs(args.input, args.output, args.neighbors, args.weights, args.metric)


def parse_args(args: list[str]) -> GridArgs | KnnArgs:
    base_parser = get_base_parser(INPUT_PATH, OUTPUT_PATH)
    parser = argparse.ArgumentParser(
        prog="models.knn",
        description="Run K-Nearest Neighbors Regressor",
    )
    grid_subparser, single_subparser = get_grid_and_single_subparsers(
        parser, base_parser
    )

    single_subparser.add_argument(
        "-n",
        "--neighbors",
        type=int,
        default="3",
        help="Number of neighbors (default: %(default)s)",
    )
    single_subparser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="distance",
        choices=["uniform", "distance"],
        help="Types of weights to use (default: %(default)s)",
    )
    single_subparser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "minkowski"],
        help="Types of metric to use (default: %(default)s)",
    )
    single_subparser.set_defaults(fun=parse_single)

    arguments = parser.parse_args(args)

    if arguments.command is None:
        parser.print_help()
        exit()

    return arguments.fun(arguments)


def run(args: KnnArgs | GridArgs) -> RegressionModelRunner:
    if isinstance(args, GridArgs):
        param_grid = json.load(args.file)

        model = GridSearchCV(
            KNeighborsRegressor(), param_grid, cv=5, scoring="neg_mean_squared_error"
        )
    else:
        model = KNeighborsRegressor(
            n_neighbors=args.neighbors, weights=args.weights, metric=args.metric
        )

    df = pd.read_csv(args.input)
    model_runner = RegressionModelRunner(df, model)

    model_runner.run()
    model_runner.save_results(args.output)
    return model_runner


def main(args: list[str]) -> None:
    run(parse_args(args))
