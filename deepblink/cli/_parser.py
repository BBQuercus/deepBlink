"""Argument parsing for deepblinks CLI."""

import argparse
import textwrap

from ..io import EXTENSIONS
from ._type import FileFolderType
from ._type import FolderType
from ._type import ShapeType


def _add_verbose(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="set program output to verbose [default: quiet]",
    )


def _parse_args_config(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for configuration."""
    parser = subparsers.add_parser(
        "config", help="\U00002699 create a configuration file for training",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="config.yaml",
        help="custom name of configuration file [default: config.yaml]",
    )
    _add_verbose(parser)
    return subparsers


def _parse_args_train(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for training."""
    parser = subparsers.add_parser(
        "train", help="\U0001F686 train a freshly baked model on a dataset",
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
    _add_verbose(parser)
    return subparsers


def _parse_args_check(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for checking."""
    parser = subparsers.add_parser(
        "check", help="\U0001F537 determine your input images' shape"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help=f"input image location [filetypes: {EXTENSIONS}]",
    )
    _add_verbose(parser)
    return subparsers


def _parse_args_predict(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for prediction."""
    parser = subparsers.add_parser(
        "predict",
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
    _add_verbose(parser)
    return subparsers


def _parse_args_create(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for creation."""
    parser = subparsers.add_parser(
        "create", help="\U0001F5BC create a new dataset from raw files"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help=f"path to raw images [filetypes: {EXTENSIONS}]",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        help="path to raw labels in csv format [default: --input/labels]",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="dataset",
        type=str,
        help="name of dataset output file. file extension is added automatically [default: 'dataset']",
    )
    _add_verbose(parser)
    return subparsers


def _parse_args_eval(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for evaluation."""
    parser = subparsers.add_parser(
        "eval", help="\U0001F3AD measure a models performance on a dataset"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )
    _add_verbose(parser)
    return subparsers


def _parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepblink",
        description=textwrap.dedent(
            """deepBlink's CLI \U0001F469\U0000200D\U0001F4BB for training, inferencing, and evaluation"""
        ),
        epilog="We hope you enjoy using deepBlink \U0001F603",
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.0.6")
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest="command")
    subparsers = _parse_args_check(subparsers)
    subparsers = _parse_args_config(subparsers)
    subparsers = _parse_args_create(subparsers)
    subparsers = _parse_args_predict(subparsers)
    subparsers = _parse_args_train(subparsers)
    # subparsers = _parse_args_eval(subparsers)

    args = parser.parse_args()
    return args
