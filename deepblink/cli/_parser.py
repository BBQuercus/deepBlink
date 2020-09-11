"""Argument parsing for deepblinks CLI."""

import argparse
import textwrap

from ._handler import EXTENSIONS
from ._type import FileFolderType
from ._type import FolderType
from ._type import ShapeType


def _parse_args_train(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for training."""
    parser = subparsers.add_parser(
        "train", help="\U0001F35E train a freshly baked model on a dataset",
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
    return subparsers


def _parse_args_check(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for checking."""
    parser = subparsers.add_parser(
        "check", help="\U0001F537 \U0001F535 determine your input images' shape"
    )
    parser.add_argument(
        "INPUT", type=str, help=f"input image location [filetypes: {EXTENSIONS}]",
    )
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
        "MODEL", type=argparse.FileType("r"), help="model .h5 file location"
    )
    parser.add_argument(
        "INPUT",
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
            """if given, uses the specified dimension arangement. otherwise falls
        back to defaults. must be in the format "(x,y,z,t,3)" using the specified
        characters."""
        ),
    )
    return subparsers


def _parse_args_eval(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for evaluation."""
    parser = subparsers.add_parser(
        "eval", help="\U0001F3AD measure a models performance on a dataset"
    )
    parser.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="set program output to verbose [default: quiet]",
    )
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest="command")
    subparsers = _parse_args_train(subparsers)
    subparsers = _parse_args_check(subparsers)
    subparsers = _parse_args_predict(subparsers)
    subparsers = _parse_args_eval(subparsers)

    args = parser.parse_args()
    return args
