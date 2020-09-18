"""Argument parsing for deepblinks CLI."""

import argparse
import textwrap

from ..io import EXTENSIONS
from ._type import FileFolderType
from ._type import FolderType
from ._type import ShapeType


def _parse_args_config(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser
):
    """Subparser for configuration."""
    parser = subparsers.add_parser(
        "config",
        parents=[parent_parser],
        add_help=False,
        description="\U00002699 Configuration submodule",
        help="\U00002699 create a configuration file for training",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="config",
        help="custom name of configuration file. file extension will be added automatically [default: config]",
    )


def _parse_args_train(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for training."""
    parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        add_help=False,
        description="\U0001F686 Training submodule",
        help="\U0001F686 train a freshly baked model on a dataset",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the experimental config.yaml file. Check the GitHub repository for an example",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help="index of GPU to be used [default: None]",
    )


def _parse_args_check(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for checking."""
    parser = subparsers.add_parser(
        "check",
        parents=[parent_parser],
        add_help=False,
        description="\U0001F537 Checking submodule",
        help="\U0001F537 determine your input images' shape",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help=f"input image location [filetypes: {EXTENSIONS}]",
    )


def _parse_args_predict(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for prediction."""
    parser = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        add_help=False,
        description="\U0001F914 Prediction submodule",
        help="\U0001F914 inference / prediction of data with a pre-trained model",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=argparse.FileType("r"),
        help="model .h5 file location",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=FolderType(),
        help="output file/folder location [default: input location]",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["csv", "txt"],
        default="csv",
        help="output file type [options: csv, txt] [default: csv]",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        default=None,
        help=textwrap.dedent(
            """if given, calculate the integrated intensity
        in the given radius around each coordinate. set radius to zero if only the
        central pixels intensity should be calculated."""
        ),
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=ShapeType(),
        default=None,
        help=textwrap.dedent(
            """if given, uses the specified dimension arrangement. otherwise falls
        back to defaults. must be in the format "(x,y,z,t,c,3)" using the specified
        characters."""
        ),
    )


def _parse_args_create(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for creation."""
    parser = subparsers.add_parser(
        "create",
        parents=[parent_parser],
        add_help=False,
        description="\U0001F5BC Creation submodule",
        help="\U0001F5BC create a new dataset from raw files",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=FolderType(),
        help=f"path to raw images [filetypes: {EXTENSIONS}]",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=FolderType(),
        help="path to raw labels in csv format [default: --input/labels]",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="dataset",
        type=str,
        help=(
            "name of dataset output file.\t"
            "file extension is added automatically.\t"
            "will be saved into the input path [default: 'dataset']"
        ),
    )


def _parse_args_eval(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for evaluation."""
    parser = subparsers.add_parser(
        "eval",
        parents=[parent_parser],
        add_help=False,
        description="\U0001F3AD Evaluation submodule",
        help="\U0001F3AD measure a models performance on a dataset",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )


def _parse_args():
    """Argument parser."""
    parent_parser = argparse.ArgumentParser(
        prog="deepblink",
        description="deepBlink's CLI \U0001F469\U0000200D\U0001F4BB for training, inferencing, and evaluation",
        epilog="We hope you enjoy using deepBlink \U0001F603",
    )
    parent_parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s 0.0.6"
    )
    parent_parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="set program output to verbose [default: quiet]",
    )

    subparsers = parent_parser.add_subparsers(dest="command", title="submodules")
    # _parse_args_eval(subparsers, parent_parser)
    _parse_args_check(subparsers, parent_parser)
    _parse_args_config(subparsers, parent_parser)
    _parse_args_create(subparsers, parent_parser)
    _parse_args_predict(subparsers, parent_parser)
    _parse_args_train(subparsers, parent_parser)

    args = parent_parser.parse_args()
    return args
