"""CLI submodule for creating a new dataset."""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from ..io import EXTENSIONS
from ..io import basename
from ..io import grab_files
from ..io import load_image
from ..util import train_valid_split
from ._parseutil import CustomFormatter
from ._parseutil import FolderType
from ._parseutil import _add_utils


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


class HandleCreate:
    """Handle creation submodule for CLI.

    Args:
        arg_input: Path to folder with images.
        arg_labels: Path to folder with labels.
        arg_name: Name of dataset file to be saved.
        logger: Logger to log verbose output.
    """

    def __init__(
        self, arg_input: str, arg_labels: str, arg_name: str, logger: logging.Logger
    ):
        self.raw_input = arg_input
        self.raw_labels = arg_labels
        self.raw_name = arg_name
        self.logger = logger
        self.logger.info("\U0001F5BC starting creation submodule")

        # TODO use inputs?
        self.test_split = 0.2
        self.valid_split = 0.2

        self.abs_input = os.path.abspath(self.raw_input)
        self.extensions = EXTENSIONS

    def __call__(self):
        """Run dataset creation."""
        image_list, label_list = self.image_label_lists
        if len(image_list) != len(label_list):
            raise ValueError(
                f"Number of images must match labels. {len(image_list)} != {len(label_list)}."
            )

        label_list_adj = [
            self.convert_labels(image, label)
            for image, label in zip(image_list, label_list)
        ]
        self.logger.debug(
            f"images converted: {len(label_list)} == {len(label_list_adj)}."
        )

        x_trainval, x_test, y_trainval, y_test = train_valid_split(
            image_list, label_list_adj, valid_split=self.test_split
        )
        x_train, x_valid, y_train, y_valid = train_valid_split(
            x_trainval, y_trainval, valid_split=self.valid_split
        )
        self.logger.info(
            f"\U0001F4A6 images split: {len(x_train)} train, {len(x_valid)} valid, {len(x_test)} test."
        )

        y_train = [y.values for y in y_train]
        y_valid = [y.values for y in y_valid]
        y_test = [y.values for y in y_test]
        np.savez_compressed(
            self.fname_out,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_test=x_test,
            y_test=y_test,
        )
        self.logger.info(f"\U0001F3C1 dataset created at {self.fname_out}.")

    @property
    def abs_labels(self):
        """Return absolute path to directory with labels."""
        if self.raw_labels is not None:
            path = os.path.abspath(self.raw_labels)
            self.logger.debug(f"using provided label path at {path}.")
        elif os.path.isdir(os.path.join(self.abs_input, "labels")):
            path = os.path.join(self.abs_input, "labels")
            self.logger.debug(f"using default label path at {path}.")
        else:
            self.logger.debug(
                f"no labels found in default {self.abs_input} or input {self.raw_labels}."
            )
            raise ValueError(
                (
                    "\U0000274C No label path found.\t"
                    "Please use a directory called 'labels' in input or use the '--labels' flag."
                )
            )
        return path

    @property
    def fname_out(self):
        """Return the absolute path to the dataset."""
        raw_name = self.raw_name
        if raw_name is not None:
            if os.path.isfile(raw_name):
                path = raw_name
                self.logger.warning(
                    f"\U000026A0 input name {raw_name} is already a file."
                )
            elif os.path.isdir(raw_name):
                path = os.path.join(raw_name, "dataset.npz")
                self.logger.warning(
                    f"\U000026A0 input name {raw_name} is already a file."
                )
            else:
                fname = raw_name[:-4] if raw_name.endswith(".npz") else raw_name
                path = os.path.join(self.abs_input, f"{fname}.npz")
                self.logger.debug(f"using given name {raw_name} only.")
        else:
            path = os.path.join(self.raw_name, "dataset.npz")
            self.logger.debug(f"using default output at {path}.")
        return path

    @property
    def image_label_lists(self):
        """Return lists with all images and labels."""
        fname_images = grab_files(self.abs_input, self.extensions)
        fname_labels = grab_files(self.abs_labels, extensions=("csv",))

        self.logger.debug(f"images - found {len(fname_images)} files: {fname_images}.")
        self.logger.debug(f"labels - found {len(fname_labels)} files: {fname_labels}.")

        images = []
        labels = []
        for image, label in zip(fname_images, fname_labels):
            if basename(image) != basename(label):
                self.logger.warning(
                    f"\U0000274C file basenames do not match! {image} != {label}."
                )
            df = pd.read_csv(label, index_col=0)
            if len(df) <= 1:
                self.logger.warning(
                    f"\U000026A0 labels for {label} empty. will not be used."
                )
                continue
            images.append(load_image(image, is_rgb=False))
            labels.append(df)

        self.logger.debug(f"using {len(images)} non-empty files.")
        return images, labels

    @staticmethod
    def convert_labels(image: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes labels to be used in deepBlink.

        Renames X/Y to c/r respectively for easier handling with rearrangement to r/c.
        Rounds coordinates on borders to prevent Fiji out-of bounds behavior.
        """
        df = df.rename(columns={"X": "c", "Y": "r"})[["r", "c"]]

        for name, var in zip(["r", "c"], image.shape):
            df[name] = df[name].where(df[name] < var, var)
            df[name] = df[name].where(df[name] > 0, 0)

        return df
