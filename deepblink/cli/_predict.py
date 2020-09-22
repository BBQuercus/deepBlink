"""CLI submodule for predicting on images."""

from typing import List
import argparse
import logging
import os
import textwrap

import numpy as np
import pandas as pd

from ..inference import get_intensities
from ..inference import predict
from ..io import EXTENSIONS
from ..io import basename
from ..io import grab_files
from ..io import load_image
from ..io import load_model
from ..util import delete_non_unique_columns
from ..util import predict_shape
from ._parseutil import CustomFormatter
from ._parseutil import FileFolderType
from ._parseutil import FolderType
from ._parseutil import ShapeType
from ._parseutil import _add_utils


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
        type=FileFolderType(EXTENSIONS),
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


class HandlePredict:
    """Handle prediction submodule for CLI.

    Args:
        arg_model: Path to model.h5 file.
        arg_input: Path to image file / folder with images.
        arg_output: Path to output directory.
        arg_radius: Size of integrated image intensity calculation.
        arg_type: Output file type.
        arg_shape: Custom shape format to label axes.
        logger: Logger to log verbose output.
    """

    def __init__(
        self,
        arg_model: str,
        arg_input: str,
        arg_output: str,
        arg_radius: int,
        arg_type: str,
        arg_shape: str,
        logger: logging.Logger,
    ):
        self.fname_model = arg_model
        self.raw_input = arg_input
        self.raw_output = arg_output
        self.radius = arg_radius
        self.type = arg_type
        self.raw_shape = arg_shape
        self.logger = logger
        self.logger.info("\U0001F914 starting prediction submodule")

        self.extensions = EXTENSIONS
        self.abs_input = os.path.abspath(self.raw_input)
        self.model = load_model(
            os.path.abspath(self.fname_model)
        )  # noqa: assignment-from-no-return
        self.logger.info("\U0001F9E0 model imported")

    def __call__(self):
        """Run prediction for all given images."""
        self.logger.info(f"\U0001F4C2 {len(self.file_list)} file(s) found")
        self.logger.info(f"\U0001F5C4{' '} output will be saved to {self.path_output}")

        for fname_in, image in zip(self.file_list, self.image_list):
            self.predict_adaptive(fname_in, image)

        self.logger.info("\U0001F3C1 all predictions are complete")

    @property
    def path_input(self) -> str:
        """Return absolute input path (dependent on file/folder input)."""
        if os.path.isdir(self.abs_input):
            path_input = self.abs_input
        elif os.path.isfile(self.abs_input):
            path_input = os.path.dirname(self.abs_input)
        return path_input

    @property
    def file_list(self) -> List[str]:
        """Return a list with all files to be processed."""
        if os.path.isdir(self.abs_input):
            file_list = grab_files(self.abs_input, self.extensions)
        elif os.path.isfile(self.abs_input):
            file_list = [self.abs_input]
        else:
            raise ImportError(
                "\U0000274C Input file(s) could not be found. Please make sure all files exist."
            )
        return file_list

    @property
    def image_list(self) -> List[np.ndarray]:
        """Return a list with all images."""
        try:
            is_rgb = "3" in self.raw_shape
        except TypeError:
            is_rgb = False
        self.logger.debug(f"loading image as RGB {is_rgb}")
        return [load_image(fname, is_rgb=is_rgb) for fname in self.file_list]

    @property
    def path_output(self) -> str:
        """Return the absolute output path (dependent if given)."""
        if os.path.exists(str(self.raw_output)):
            outpath = os.path.abspath(self.raw_output)
        else:
            outpath = self.path_input
        return outpath

    # TODO solve double definition of replace_chars here and in ShapeType
    # TODO solve mypy return type bug
    @property
    def shape(self):
        """Resolve input shape."""
        first_image = self.image_list[0]
        if not all([i.ndim == first_image.ndim for i in self.image_list]):
            raise ValueError("Images must all have the same number of dimensions.")
        if not all([i.shape == first_image.shape for i in self.image_list]):
            self.logger.warning(
                "\U000026A0 images do not have equal shapes (dimensions match)"
            )

        if self.raw_shape is None:
            shape = predict_shape(self.image_list[0].shape)
            self.logger.info(f"\U0001F535 using predicted shape of {shape}")
        else:
            shape = self.raw_shape
            self.logger.info(f"\U0001F535 using provided input shape of {shape}")
        for c in ["(", ")", " "]:
            shape = shape.replace(c, "")
        shape = shape.split(",")
        return shape

    def save_output(self, fname_in: str, df: pd.DataFrame) -> None:
        """Save coordinate list to file with appropriate header."""
        fname_out = os.path.join(self.path_output, f"{basename(fname_in)}.{self.type}")
        df = delete_non_unique_columns(df)
        self.logger.debug(f"non-unique columns to be saved are {df.columns}")

        if self.type == "txt":
            header = " ".join(df.columns.to_list())
            np.savetxt(
                fname_out,
                df.values,
                fmt="%.4f",
                delimiter=" ",
                header=header,
                comments="",
            )
        if self.type == "csv":
            df.to_csv(fname_out, index=False)
        self.logger.info(
            f"\U0001F3C3 prediction of file {fname_in} saved as {fname_out}"
        )

    def predict_single(
        self, image: np.ndarray, c_idx: int, t_idx: int, z_idx: int
    ) -> pd.DataFrame:
        """Predict a single (x,y) image at given c, t, z positions."""
        coords = predict(image, self.model)
        df = pd.DataFrame(coords, columns=["y", "x"])  # originally r, c
        df["c"] = c_idx
        df["t"] = t_idx
        df["z"] = z_idx
        if self.radius is not None:
            df["i"] = get_intensities(image, coords, self.radius)
        return df

    def predict_adaptive(self, fname_in: str, image: np.ndarray) -> None:
        """Predict and save a single image."""
        order = ["c", "t", "z", "y", "x"]
        shape = self.shape

        # Create an image and shape with all possible dimensions
        for i in order:
            if i not in shape:
                image = np.expand_dims(image, axis=-1)
                shape.append(i)

        # Rearange all axes to match to the desired order
        for destination, name in enumerate(order):
            source = shape.index(name)
            image = np.moveaxis(image, source, destination)
            shape.insert(destination, shape.pop(source))
        if not order == shape:
            self.logger.debug("axes rearangement did not work properly")

        # Iterate through c, t, and z
        df = pd.DataFrame()
        for c_idx, t_ser in enumerate(image):
            for t_idx, z_ser in enumerate(t_ser):
                for z_idx, single_image in enumerate(z_ser):
                    curr_df = self.predict_single(single_image, c_idx, t_idx, z_idx)
                    df = df.append(curr_df)

        self.logger.debug(f"completed prediction loop with\n{df.head()}")
        self.save_output(fname_in, df)
