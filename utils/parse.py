import argparse
import inspect
import os
from dataclasses import dataclass
from pathlib import Path


############################## BASE PARSING ##############################
@dataclass
class BaseArgs:
    input: Path
    output: Path


def get_base_parser(
    default_in_path: str, default_out_path: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group(title="I/O files argument handling")
    group.add_argument(
        "-i",
        # "--input",
        type=parse_input_path,
        help="Input file path (default: %(default)s)",
        default=default_in_path,
        metavar="INPUT_PATH",
        dest="input",
    )
    group.add_argument(
        "-o",
        # "--output",
        type=parse_output_path,
        help="Output file path (default: %(default)s)",
        default=default_out_path,
        metavar="OUTPUT_PATH",
        dest="output",
    )
    return parser


def parse_input_path(_path) -> Path:
    path = Path(_path)
    assert path.exists(), f"Provided path '{path}' doesn't exist!"
    return path


def parse_output_path(_path) -> Path:
    path = Path(_path)
    if not path.parent.exists():
        user_input = input(
            f"Provided output parent(s) dir(s) '{path.parent}' do not exist,\nDo you want to create them? [Y/n]: "
        )
        if user_input.lower().strip().replace("y", "") == "":
            path.parent.mkdir(parents=True)
        else:
            print("Exiting ...")
            exit()
    return path


############################## MODEL PARSING ##############################
@dataclass
class GridArgs(BaseArgs):
    file: str


def parse_grid(args: argparse.Namespace) -> GridArgs:
    return GridArgs(args.input, args.output, args.file)


def get_grid_and_single_subparsers(
    parser: argparse.ArgumentParser,
    base_parser: argparse.ArgumentParser,
) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    # Complicated trick to know where this is been called from
    _from_file = inspect.currentframe().f_back.f_globals["__file__"]
    # Add subparsers to parser
    subparsers = parser.add_subparsers(dest="command")
    # Grid suparser
    grid_subparser = subparsers.add_parser(
        "grid",
        description="Perform a search over parameter grid to find best parameters for the model",
        parents=[base_parser],
    )
    grid_subparser.add_argument(
        "-f",
        "--file",
        type=argparse.FileType("r"),
        default=os.path.join(os.path.dirname(_from_file), "params.json"),
        help=".json file with parameter grid",
    )
    grid_subparser.set_defaults(fun=parse_grid)

    # Single suparser
    single_subparser = subparsers.add_parser(
        "single",
        description="Fit model and make predictions with given hyper-parameters",
        parents=[base_parser],
    )

    return (grid_subparser, single_subparser)
