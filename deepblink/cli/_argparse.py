"""CLI argument parsing."""

import argparse

# from ..io import EXTENSIONS
from ._parseutil import Color
from ._parseutil import CustomFormatter
from ._parseutil import FileFolderType
from ._parseutil import FileType
from ._parseutil import FolderType
from ._parseutil import ShapeType
from ._parseutil import _add_utils

EXTENSIONS = ("tif", "jpeg", "jpg", "png")


def _parse_args_check(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for checking."""
    parser = subparsers.add_parser(
        "check",
        parents=[parent_parser],
        formatter_class=CustomFormatter,
        add_help=False,
        description=(
            f"\U0001F537 {Color.title}Checking submodule{Color.end} \U0001F537\n\n"
            "Check the arrangement of your image's axis also known as image shape. "
        ),
        help="\U0001F537 Determine your input image's shape.",
    )
    group1 = parser.add_argument_group(f"{Color.required}Required{Color.end}")
    group1.add_argument(
        "INPUT",
        type=FileType(EXTENSIONS),
        help=(
            "Input image. "
            "Path to the image file to be checked. "
            "The path be relative (e.g. ../dir) or absolute (e.g. /Users/myname/). "
            "Input can either be given as path to a directory containing files or as a single file. "
            "Note that only the specified filetypes will be processed. "
            f"[required] [filetypes: {', '.join(EXTENSIONS)}]"
        ),
    )
    _add_utils(parser)


def _parse_args_config(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser
):
    """Subparser for configuration."""
    parser = subparsers.add_parser(
        "config",
        parents=[parent_parser],
        add_help=False,
        formatter_class=CustomFormatter,
        description=(
            f"\U0001F528 {Color.title}Configuration submodule{Color.end} \U0001F528\n\n"
            "Prepare a configuration file used to adjust parameters during training. "
        ),
        help="\U0001F528 Create a configuration file for training.",
    )
    group2 = parser.add_argument_group(f"{Color.optional}Optional{Color.end}")
    group2.add_argument(
        "-n",
        "--name",
        type=str,
        default="config",
        help=(
            "Custom configuration name. "
            'The file extension "yaml" will be added automatically to the given name. '
            '[default: "config"]'
        ),
    )
    group2.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help=(
            "Save configuration file without description of values. "
            "Shorter but not descriptive."
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
        description=(
            f"\U0001F4BE {Color.title}Creation submodule{Color.end} \U0001F4BE\n\n"
            "Create a custom dataset with raw files and corresponding labels. "
            "Relies on labeling output from FIJI that was saved with the provided macro "
            "or the standard TrackMate coordinate output. "
            'Both are described here "https://github.com/BBQuercus/deepBlink/wiki/Datasets".'
        ),
        help="\U0001F4BE Create a new dataset from raw files.",
    )
    group1 = parser.add_argument_group(f"{Color.required}Required{Color.end}")
    group1.add_argument(
        "-i",
        "--input",
        required=True,
        type=FolderType(),
        help=(
            "Path to the directory containing raw images. "
            "Note that only the specified filetypes will be processed. "
            f"[required] [filetypes: {', '.join(EXTENSIONS)}]"
        ),
    )
    group2 = parser.add_argument_group(f"{Color.optional}Optional{Color.end}")
    group2.add_argument(
        "-l",
        "--labels",
        type=FolderType(),
        help=(
            "Path to the directory containing labels in csv format. "
            "The default path accounts for using the FIJI macro described on the wiki. "
            "[default: --INPUT/labels/]"
        ),
    )
    group2.add_argument(
        "-n",
        "--name",
        default="dataset",
        type=str,
        help=(
            "Custom dataset name. "
            'The file extension "npz" will be added automatically. '
            '[default: "dataset"]'
        ),
    )
    group2.add_argument(
        "-s",
        "--size",
        default=512,
        type=int,
        help=(
            "Image crop size. "
            "If given, crops all images into the specified size. "
            "Will crop non-overlapping and ignore areas that did not get covered."
            "deepBlink requires square images powers of 2, such as 256, 512... "
            "[default: 512]"
        ),
    )
    group2.add_argument(
        "-vs",
        "--validsplit",
        default=0.2,
        type=float,
        help=(
            "Validation split. "
            "Split percentage (scaled between 0 - 1) of validation vs. train set. "
            "Note the validation split is done after splitting test and trainval. "
            "[default: 0.2]"
        ),
    )
    group2.add_argument(
        "-ts",
        "--testsplit",
        default=0.2,
        type=float,
        help=(
            "Testing split. "
            "Split percentage (scaled between 0 - 1) of test vs. trainval set. "
            "[default: 0.2]"
        ),
    )
    _add_utils(parser)


def _parse_args_download(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for downloading."""
    parser = subparsers.add_parser(
        "download",
        parents=[parent_parser],
        formatter_class=CustomFormatter,
        add_help=False,
        description=(
            f"\U0001F4E5 {Color.title}Downloading submodule{Color.end} \U0001F4E5\n\n"
            "Download pre-trained models from our online figshare repository to predict. "
        ),
        help="\U0001F4E5 Download pre-trained models for use.",
    )
    group2 = parser.add_argument_group(f"{Color.optional}Optional{Color.end}")
    group2.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help=(
            "Input name. "
            "Name of the model to be downloaded. "
            'Note that only the models listed in "deepblink download --list" will be processed. '
            "[default: None]"
        ),
    )
    group2.add_argument(
        "-l",
        "--list",
        action="store_true",
        help=("List available models. " "Name of the model to be downloaded. "),
    )
    group2.add_argument(
        "-a",
        "--all",
        action="store_true",
        help=(
            "Download all available models. "
            "If passed, all models will be downloaded. "
        ),
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
        description=(
            f"\U0001F914 {Color.title}Prediction submodule{Color.end} \U0001F914\n\n"
            "Use a pre-trained model to predict blob coordinates on new data. "
            "In addition to the required model and input file or folder, "
            "several optional features are accessible as described below."
        ),
        help="\U0001F914 Predict on data with a pre-trained model.",
    )
    group1 = parser.add_argument_group(f"{Color.required}Required{Color.end}")
    group1.add_argument(
        "-i",
        "--input",
        required=True,
        type=FileFolderType(EXTENSIONS),
        help=(
            "Image files to predict on. "
            "Input can either be given as path to a directory containing files or as a single file. "
            "The path be relative (e.g. ../dir) or absolute (e.g. /Users/myname/). "
            "Fileglobs are currently not available. "
            "Note that only the specified filetypes will be processed. "
            f"[required] [filetypes: {', '.join(EXTENSIONS)}]"
        ),
    )
    group1.add_argument(
        "-m",
        "--model",
        required=True,
        type=FileType(["h5"]),
        help=(
            "DeepBlink model. "
            'Model has to be of file type ".h5". '
            'The path can be relative or absolute as described in "--input". '
            'Model can either be trained on custom data using "deepblink train" or using a pre-trained '
            'model available through the GitHub wiki on "https://github.com/BBQuercus/deepBlink/wiki". '
            "[required]"
        ),
    )
    group2 = parser.add_argument_group(f"{Color.optional}Optional{Color.end}")
    group2.add_argument(
        "-o",
        "--output",
        type=FolderType(),
        help=(
            "Output folder path. "
            "Path to the directory into which all output files are saved. "
            "Output files will automatically take the same name as their corresponding image. "
            "[default: input location]"
        ),
    )
    group2.add_argument(
        "-r",
        "--radius",
        type=int,
        default=None,
        help=(
            "Intensity radius. "
            "If given, will calculate the integrated intensity in the specified radius around each coordinate. "
            "If the radius is set to zero if only the central pixels intensity should be calculated. "
            'The intensity is added as additional column to the output file called "i". '
            "[default: None]"
        ),
    )
    group2.add_argument(
        "-s",
        "--shape",
        type=ShapeType(),
        default=None,
        help=(
            "Image shape."
            "Used to assess the arrangement of input image axes otherwise known as shape. "
            "If not given, uses a basic prediction based on common defaults. "
            'Must be in the format "(x,y,z,t,c,3)" using the specified characters. '
            'If unsure, use "deepblink check" to determine your images shape '
            "and more detailed information. "
            "[default: None]"
        ),
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
        description=(
            f"\U0001F686 {Color.title}Training submodule{Color.end} \U0001F686\n\n"
            'Train a custom model using a custom dataset created in "deepblink create" '
            "or using a published dataset."
        ),
        help="\U0001F686 Train a freshly baked model on a dataset.",
    )
    group1 = parser.add_argument_group(f"{Color.required}Required{Color.end}")
    group1.add_argument(
        "-c",
        "--config",
        type=FileType(["yaml"]),
        required=True,
        help=(
            "Configuration file. "
            'Path to the config.yaml created using "deepblink config". '
            "The path be relative (e.g. ../dir) or absolute (e.g. /Users/myname/). "
            "Please see the training information on the wiki to configure the file to your requirements. "
            "[required]"
        ),
    )
    group2 = parser.add_argument_group(f"{Color.optional}Optional{Color.end}")
    group2.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help=(
            "GPU index. "
            "Value passed CUDA_VISIBLE_DEVICES if a GPU is used for training. "
            "[default: None]"
        ),
    )
    _add_utils(parser)
