"""CLI submodule for creating a new dataset."""

from typing import List, Tuple, Union
import logging
import os

import numpy as np
import pandas as pd
import skimage.util

from ..io import EXTENSIONS
from ..io import basename
from ..io import grab_files
from ..io import load_image
from ..util import predict_pixel_size
from ..util import train_valid_split


class HandleCreate:
    """Handle creation submodule for CLI.

    Args:
        arg_input: Path to folder with images.
        arg_labels: Path to folder with labels.
        arg_name: Name of dataset file to be saved.
        arg_size: Size of image to be cropped.
        arg_pixel_size: Pixel size of images.
        arg_testsplit: Test vs. Trainval split percentage.
        arg_testsplit: Valid vs. Train split percentage.
        logger: Logger to log verbose output.
    """

    # pylint: disable=too-many-instance-attributes
    # We require many attributes for the splits

    def __init__(
        self,
        arg_input: str,
        arg_labels: str,
        arg_name: str,
        arg_size: int,
        arg_pixel_size: Union[float, Tuple[float, float]],
        arg_testsplit: int,
        arg_validsplit: int,
        arg_minspots: int,
        logger: logging.Logger,
    ):
        self.raw_input = arg_input
        self.raw_labels = arg_labels
        self.raw_name = arg_name
        self.img_size = arg_size
        self.pixel_size = arg_pixel_size
        self.test_split = arg_testsplit
        self.valid_split = arg_validsplit
        self.minspots = max(1, arg_minspots)
        self.logger = logger
        self.logger.info("\U0001F5BC starting creation submodule")

        self.abs_input = os.path.abspath(self.raw_input)
        self.extensions = EXTENSIONS

    def __call__(self):
        """Run dataset creation."""
        self.get_file_lists()
        self.train_val_test_split()
        self.save_npz()
        self.logger.info(f"\U0001F3C1 dataset created at {self.fname_out}")

    @property
    def abs_labels(self) -> str:
        """Return absolute path to directory with labels."""
        # Full path name
        if self.raw_labels is not None:
            path = os.path.abspath(self.raw_labels)
            self.logger.debug(f"using provided label path at {path}")
        # Default path in input directory
        elif os.path.isdir(self.abs_input):
            path = self.abs_input
            self.logger.debug(f"using default label path at {path}")
        else:
            self.logger.debug(
                f"no labels found in default {self.abs_input} or input {self.raw_labels}"
            )
            raise ValueError(
                (
                    "\U0000274C No label path found.\t"
                    "Please provide labels in the input directory or use the '--labels' flag."
                )
            )
        return path

    @property
    def fname_out(self) -> str:
        """Return the absolute path to the dataset."""
        if self.raw_name is not None:
            # Full path and file given
            if os.path.isfile(self.raw_name):
                path = self.raw_name
                self.logger.warning(
                    f"\U000026A0 input name {self.raw_name} is already a file"
                )
            # Full path only
            elif os.path.isdir(self.raw_name):
                path = os.path.join(self.raw_name, "dataset.npz")
                self.logger.warning(
                    f"\U000026A0 input name {self.raw_name} is already a file"
                )
            # Name only
            else:
                fname = (
                    self.raw_name[:-4]
                    if self.raw_name.endswith(".npz")
                    else self.raw_name
                )
                path = os.path.join(os.getcwd(), f"{fname}.npz")
                self.logger.debug(f"using given name {self.raw_name}")
        else:
            path = os.path.join(self.raw_name, "dataset.npz")
            self.logger.debug(f"using default output at {path}")
        return path

    def get_pixel_size(self, image: str) -> Tuple[float, float]:
        """Return the pixel size of an image."""
        # Use user-provided pixel size
        if self.pixel_size is not None:
            self.logger.debug(f"using provided pixel size {self.pixel_size}")
            if isinstance(self.pixel_size, tuple):
                return self.pixel_size
            return (self.pixel_size, self.pixel_size)

        # Try to predict pixel size
        try:
            size = predict_pixel_size(image)
            self.logger.debug(f"using predicted pixel size {size}")
            return size
        except (ValueError, KeyError, IndexError) as e:
            self.logger.warning(
                f"\U000026A0 pixel size for image {image} could not be predicted."
            )

        self.logger.debug(f"defaulting pixel size to {size}")
        return (1.0, 1.0)

    @property
    def image_label_size_lists(
        self,
    ) -> Tuple[List[np.ndarray], List[pd.DataFrame], List[Tuple[float, float]]]:
        """Return lists with all images, labels, and sizes."""
        fname_images = grab_files(self.abs_input, self.extensions)
        fname_labels = grab_files(self.abs_labels, extensions=("csv",))
        self.logger.debug(f"images - found {len(fname_images)} files: {fname_images}")
        self.logger.debug(f"labels - found {len(fname_labels)} files: {fname_labels}")

        images = []
        labels = []
        sizes = []
        for image, label in zip(fname_images, fname_labels):
            if basename(image) != basename(label):
                self.logger.warning(
                    f"\U0000274C file basenames do not match! {image} != {label}"
                )

            # Read label
            try:
                df = pd.read_csv(label, index_col=0)
            except pd.errors.EmptyDataError:
                self.logger.warning(f"\U0000274C label empty! {label} won't be used")
                continue
            if len(df) <= 1:
                self.logger.warning(
                    f"\U000026A0 labels for {label} empty. will not be used"
                )
                continue

            # Read image
            img = load_image(image, is_rgb=False)
            if not all(shape >= self.img_size for shape in img.shape):
                self.logger.warning(
                    f"\U000026A0 image {image} was too small! {img.shape} < {self.img_size}"
                )
                continue

            size = self.get_pixel_size(image)
            images.append(img)
            labels.append(df)
            sizes.append(size)

        if not images:
            raise ValueError(
                "No images matched the format criteria. Please check image size and labelling."
            )
        self.logger.debug(f"using {len(images)} non-empty files")
        return images, labels, sizes

    @staticmethod
    def convert_labels(
        image: np.ndarray, df: pd.DataFrame, pixel_size: Tuple[float, float]
    ) -> pd.DataFrame:
        """Pre-processes labels to be used in deepBlink.

        Renames X/Y to c/r respectively for easier handling with rearrangement to r/c.
        Rounds coordinates on borders to prevent Fiji out-of bounds behavior.
        """
        # Fiji point label format
        if all(c in df.columns for c in ("X", "Y")):
            df = df.rename(columns={"X": "c", "Y": "r"})[["r", "c"]]
        # TrackMate export format
        elif all(c in df.columns for c in ("POSITION_X", "POSITION_Y")):
            df = df[~df.index.isna()]  # Remove unused headers for TrackMate v7.0.0+
            df = df.rename(columns={"POSITION_X": "c", "POSITION_Y": "r"})[["r", "c"]]
            df = df.reset_index(drop=True)
        else:
            raise ValueError(
                "Format of input labels not recognized. "
                "Requires X,Y or POSITION_X,POSITION_Y in columns. "
                f"Columns found are: {df.columns.to_list()}."
            )

        # Clip upper and lower bounds of coordinates
        df = df.astype({"r": np.float64, "c": np.float64})
        for name, var in zip(["r", "c"], image.shape):
            df[name] = df[name].where(df[name] < var, var)
            df[name] = df[name].where(df[name] > 0, 0)

        # Scale coordinates to pixel size
        size_x, size_y = pixel_size
        df["r"] = df["r"] / size_y
        df["c"] = df["c"] / size_x
        return df

    def crop_image(
        self, image: np.ndarray, df: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
        """Crop a image / label pair to a uniform size and scale labels accordingly."""
        if self.img_size is None:
            self.logger.debug(f"using unchanged size in {image.shape}")
            return [image], [df]

        if any([self.img_size > s for s in image.shape]):
            self.logger.debug(f"skipping image due to small size {image.shape}")
            return [], []

        windows = skimage.util.view_as_windows(
            image, window_shape=(self.img_size, self.img_size), step=self.img_size
        )

        img_list = []
        df_list = []
        for r, cimages in enumerate(windows):
            for c, img in enumerate(cimages):
                # Extract coordinates from crop
                r_min = r * self.img_size
                c_min = c * self.img_size
                df_slice = df.loc[
                    (
                        (df["r"] >= r_min)
                        & (df["r"] <= r_min + self.img_size)
                        & (df["c"] >= c_min)
                        & (df["c"] <= c_min + self.img_size)
                    )
                ]
                df_norm = df_slice.copy()
                df_norm.loc[:, "r"] = df_slice["r"] - r_min
                df_norm.loc[:, "c"] = df_slice["c"] - c_min

                if len(df_norm) >= self.minspots:
                    img_list.append(img)
                    df_list.append(df_norm)

        self.logger.debug(
            f"converted original size of {image.shape} into {len(df_list)} crops"
        )
        return img_list, df_list

    def get_file_lists(self):
        """Create a file list after cropping over-sized images."""
        self.image_list = []
        self.label_list = []

        for image, label, size in zip(*self.image_label_size_lists):
            label_normalized = self.convert_labels(
                image=image, df=label, pixel_size=size
            )
            image_crops, label_crops = self.crop_image(image=image, df=label_normalized)
            self.image_list.extend(image_crops)
            self.label_list.extend(label_crops)

        self.logger.debug(
            f"images converted: {len(self.image_list)} == {len(self.label_list)}"
        )

    def train_val_test_split(self):
        """Two-step split of the input data into train/valid/test."""
        x_trainval, self.x_test, y_trainval, y_test = train_valid_split(
            x_list=self.image_list, y_list=self.label_list, valid_split=self.test_split
        )
        self.x_train, self.x_valid, y_train, y_valid = train_valid_split(
            x_list=x_trainval, y_list=y_trainval, valid_split=self.valid_split
        )
        self.logger.info(
            f"\U0001F4A6 images split: {len(self.x_train)} train, {len(self.x_valid)} valid, {len(self.x_test)} test"
        )

        # Convert DataFrame to Numpy array
        self.y_train = np.array([y.values for y in y_train], dtype=object)
        self.y_valid = np.array([y.values for y in y_valid], dtype=object)
        self.y_test = np.array([y.values for y in y_test], dtype=object)

    def save_npz(self):
        """Save dataset splits as single npz file."""
        np.savez_compressed(
            self.fname_out,
            x_train=self.x_train,
            y_train=self.y_train,
            x_valid=self.x_valid,
            y_valid=self.y_valid,
            x_test=self.x_test,
            y_test=self.y_test,
        )
