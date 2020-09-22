"""CLI submodule for creating a new dataset."""

from typing import List, Tuple
import argparse
import logging
import os

import numpy as np
import pandas as pd
import skimage.util

from ..io import EXTENSIONS
from ..io import basename
from ..io import grab_files
from ..io import load_image
from ..util import train_valid_split
from ._parseutil import CustomFormatter
from ._parseutil import FMT
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
        description=(
            f"\U0001F4BE {FMT.dc}Creation submodule{FMT.e} \U0001F4BE\n\n"
            "Create a custom dataset with raw files and corresponding labels. "
            "Relies on labeling output from FIJI that was saved with the provided macro "
            f"described here {FMT.u}https://github.com/BBQuercus/deepBlink/wiki/Datasets{FMT.e}."
        ),
        help="\U0001F4BE Create a new dataset from raw files.",
    )
    group1 = parser.add_argument_group(f"{FMT.r}Required{FMT.e}")
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
    group2 = parser.add_argument_group(f"{FMT.g}Optional{FMT.e}")
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
        default=None,
        type=int,
        help=(
            "Image crop size. "
            "If given, crops all images into the specified size. "
            "Will crop non-overlapping and ignore areas that did not get covered."
            "deepBlink requires square images powers of 2, such as 256, 512... "
            "[default: None]"
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


class HandleCreate:
    """Handle creation submodule for CLI.

    Args:
        arg_input: Path to folder with images.
        arg_labels: Path to folder with labels.
        arg_name: Name of dataset file to be saved.
        arg_size: Size of image to be cropped.
        arg_testsplit: Test vs. Trainval split percentage.
        arg_testsplit: Valid vs. Train split percentage.
        logger: Logger to log verbose output.
    """

    def __init__(
        self,
        arg_input: str,
        arg_labels: str,
        arg_name: str,
        arg_size: int,
        arg_testsplit: int,
        arg_validsplit: int,
        logger: logging.Logger,
    ):
        self.raw_input = arg_input
        self.raw_labels = arg_labels
        self.raw_name = arg_name
        self.img_size = arg_size
        self.test_split = arg_testsplit
        self.valid_split = arg_validsplit
        self.logger = logger
        self.logger.info("\U0001F5BC starting creation submodule")

        self.abs_input = os.path.abspath(self.raw_input)
        self.extensions = EXTENSIONS

    def __call__(self):
        """Run dataset creation."""
        image_list = []
        label_list = []
        for image, label in zip(*self.image_label_lists):
            label_norm = self.convert_labels(image=image, df=label)
            image_crops, label_crops = self.crop_images(image=image, df=label_norm)
            image_list.extend(image_crops)
            label_list.extend(label_crops)
        self.logger.debug(f"images converted: {len(image_list)} == {len(label_list)}")

        x_trainval, x_test, y_trainval, y_test = train_valid_split(
            image_list, label_list, valid_split=self.test_split
        )
        x_train, x_valid, y_train, y_valid = train_valid_split(
            x_trainval, y_trainval, valid_split=self.valid_split
        )
        self.logger.info(
            f"\U0001F4A6 images split: {len(x_train)} train, {len(x_valid)} valid, {len(x_test)} test"
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
        self.logger.info(f"\U0001F3C1 dataset created at {self.fname_out}")

    @property
    def abs_labels(self):
        """Return absolute path to directory with labels."""
        if self.raw_labels is not None:
            path = os.path.abspath(self.raw_labels)
            self.logger.debug(f"using provided label path at {path}")
        elif os.path.isdir(os.path.join(self.abs_input, "labels")):
            path = os.path.join(self.abs_input, "labels")
            self.logger.debug(f"using default label path at {path}")
        else:
            self.logger.debug(
                f"no labels found in default {self.abs_input} or input {self.raw_labels}"
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
                    f"\U000026A0 input name {raw_name} is already a file"
                )
            elif os.path.isdir(raw_name):
                path = os.path.join(raw_name, "dataset.npz")
                self.logger.warning(
                    f"\U000026A0 input name {raw_name} is already a file"
                )
            else:
                fname = raw_name[:-4] if raw_name.endswith(".npz") else raw_name
                path = os.path.join(self.abs_input, f"{fname}.npz")
                self.logger.debug(f"using given name {raw_name}")
        else:
            path = os.path.join(self.raw_name, "dataset.npz")
            self.logger.debug(f"using default output at {path}")
        return path

    @property
    def image_label_lists(self):
        """Return lists with all images and labels."""
        fname_images = grab_files(self.abs_input, self.extensions)
        fname_labels = grab_files(self.abs_labels, extensions=("csv",))

        self.logger.debug(f"images - found {len(fname_images)} files: {fname_images}")
        self.logger.debug(f"labels - found {len(fname_labels)} files: {fname_labels}")

        images = []
        labels = []
        for image, label in zip(fname_images, fname_labels):
            if basename(image) != basename(label):
                self.logger.warning(
                    f"\U0000274C file basenames do not match! {image} != {label}"
                )
            df = pd.read_csv(label, index_col=0)
            if len(df) <= 1:
                self.logger.warning(
                    f"\U000026A0 labels for {label} empty. will not be used"
                )
                continue
            images.append(load_image(image, is_rgb=False))
            labels.append(df)

        if len(images) != len(labels):
            raise ValueError(
                f"Number of images and labels must match. {len(images)} != {len(labels)}."
            )
        self.logger.debug(f"using {len(images)} non-empty files")

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

    def crop_images(
        self, image: np.ndarray, df: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
        """Crop images to a uniform size and scale labels accordingly."""
        size = self.img_size
        if size is None:
            self.logger.debug(f"using unchanged size in {image.shape}")
            return [image], [df]

        windows = skimage.util.view_as_windows(
            image, window_shape=(size, size), step=size
        )

        img_list = []
        df_list = []
        for r, cimages in enumerate(windows):
            for c, img in enumerate(cimages):
                img_list.append(img)

                r_min = r * size
                c_min = c * size
                df_slice = df.loc[
                    (
                        (df["r"] >= r_min)
                        & (df["r"] <= r_min + size)
                        & (df["c"] >= c_min)
                        & (df["c"] <= c_min + size)
                    )
                ]
                df_norm = df_slice.copy()
                df_norm.loc[:, "r"] = df_slice["r"] - r_min
                df_norm.loc[:, "c"] = df_slice["c"] - c_min
                df_list.append(df_norm)

        self.logger.debug(
            f"converted original size of {image.shape} into {len(df_list)} crops"
        )
        return img_list, df_list
