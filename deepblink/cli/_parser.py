"""Argument parsing for deepblinks CLI."""

import argparse
import textwrap

from ..io import EXTENSIONS
from ._parseutil import FileFolderType
from ._parseutil import FolderType
from ._parseutil import ShapeType


def _parse_args_config(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser
):
    """Subparser for configuration."""
    parser = subparsers.add_parser(
        "config",
        parents=[parent_parser],
        add_help=False,
        formatter_class=CustomFormatter,
        description="\U00002699 Configuration submodule",
        help="\U00002699 create a configuration file for training",
    )
    group2 = parser.add_argument_group("Optional")
    group2.add_argument(
        "-n",
        "--name",
        type=str,
        default="config",
        help="custom name of configuration file. file extension will be added automatically [default: config]",
    )
    _add_utils(parser)


def _parse_args_train(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for training."""
    parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        formatter_class=CustomFormatter,
        add_help=False,
        description="\U0001F686 Training submodule",
        help="\U0001F686 train a freshly baked model on a dataset",
    )
    group1 = parser.add_argument_group("Required")
    group1.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the experimental config.yaml file. Check the GitHub repository for an example [required]",
    )
    group2 = parser.add_argument_group("Optional")
    group2.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help="index of GPU to be used [default: None]",
    )
    _add_utils(parser)


def _parse_args_check(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for checking."""
    parser = subparsers.add_parser(
        "check",
        parents=[parent_parser],
        formatter_class=CustomFormatter,
        add_help=False,
        description="\U0001F537 Checking submodule",
        help="\U0001F537 determine your input images' shape",
    )
    group1 = parser.add_argument_group("Required")
    group1.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help=f"input image location [required] [filetypes: {EXTENSIONS}]",
    )
    _add_utils(parser)


def _parse_args_predict(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for prediction."""
    parser = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        add_help=False,
        formatter_class=CustomFormatter,
        description="\U0001F914 Prediction submodule",
        help="\U0001F914 inference / prediction of data with a pre-trained model",
    )
    group1 = parser.add_argument_group("Required")
    group1.add_argument(
        "-i",
        "--input",
        required=True,
        type=FileFolderType(),
        help=f"input file/folder location [required] [filetypes: {EXTENSIONS}]",
    )
    group1.add_argument(
        "-m",
        "--model",
        required=True,
        type=argparse.FileType("r"),
        help="model .h5 file location [required]",
    )
    group2 = parser.add_argument_group("Optional")
    group2.add_argument(
        "-o",
        "--output",
        type=FolderType(),
        help="output file/folder location [default: input location]",
    )
    group2.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["csv", "txt"],
        default="csv",
        help="output file type [options: csv, txt] [default: csv]",
    )
    group2.add_argument(
        "-r",
        "--radius",
        type=int,
        default=None,
        help=textwrap.dedent(
            """if given, calculate the integrated intensity
        in the given radius around each coordinate. set radius to zero if only the
        central pixels intensity should be calculated"""
        ),
    )
    group2.add_argument(
        "-s",
        "--shape",
        type=ShapeType(),
        default=None,
        help=textwrap.dedent(
            """if given, uses the specified dimension arrangement. otherwise falls
        back to defaults. must be in the format "(x,y,z,t,c,3)" using the specified
        characters"""
        ),
    )
    _add_utils(parser)


def _parse_args_create(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for creation."""
    parser = subparsers.add_parser(
        "create",
        parents=[parent_parser],
        formatter_class=CustomFormatter,
        add_help=False,
        description="\U0001F5BC Creation submodule",
        help="\U0001F5BC create a new dataset from raw files",
    )
    group1 = parser.add_argument_group("Required")
    group1.add_argument(
        "-i",
        "--input",
        required=True,
        type=FolderType(),
        help=f"path to raw images [required] [filetypes: {EXTENSIONS}]",
    )
    group2 = parser.add_argument_group("Optional")
    group2.add_argument(
        "-l",
        "--labels",
        type=FolderType(),
        help="path to raw labels in csv format [default: --input/labels]",
    )
    group2.add_argument(
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
    _add_utils(parser)


class CustomFormatter(argparse.HelpFormatter):
    """Custom changes to argparse's default help text formatter."""

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = "Usage: "
        return super(CustomFormatter, self).add_usage(usage, actions, groups, prefix)


# TODO find a simpler and safer solution
def _add_utils(parser: argparse.ArgumentParser):
    """A very hacky way of trying to move this group to the bottom of help text."""
    group = parser.add_argument_group("General utilities")
    group.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    group.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s 0.0.6",
        help="show %(prog)s's version number and exit",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="set program output to verbose [default: quiet]",
    )
    group.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)


def _parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepblink",
        description="deepBlink's CLI \U0001F469\U0000200D\U0001F4BB for training, inferencing, and evaluation",
        epilog="We hope you enjoy using deepBlink \U0001F603",
        formatter_class=CustomFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(title="Commands", dest="command")
    parent_parser = argparse.ArgumentParser(add_help=False)
    _parse_args_check(subparsers, parent_parser)
    _parse_args_config(subparsers, parent_parser)
    _parse_args_create(subparsers, parent_parser)
    _parse_args_predict(subparsers, parent_parser)
    _parse_args_train(subparsers, parent_parser)
    _add_utils(parser)

    args = parser.parse_args()
    return args
