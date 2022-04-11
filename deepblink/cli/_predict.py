"""CLI submodule for predicting on images."""

from typing import List, Tuple, Union
import logging
import os

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
from ._util import get_pixel_size


class HandlePredict:
    """Handle prediction submodule for CLI.

    Args:
        arg_model: Path to model.h5 file.
        arg_input: Path to image file / folder with images.
        arg_output: Path to output directory.
        arg_radius: Size of integrated image intensity calculation.
        arg_shape: Custom shape format to label axes.
        arg_probability: Probability threshold for prediction.
        arg_pixel_size: Pixel size of image.
        logger: Logger to log verbose output.
    """

    def __init__(
        self,
        arg_model: str,
        arg_input: str,
        arg_output: str,
        arg_radius: int,
        arg_shape: str,
        arg_probability: float,
        arg_pixel_size: Union[float, Tuple[float, float]],
        logger: logging.Logger,
    ):
        self.fname_model = arg_model
        self.raw_input = arg_input
        self.raw_output = arg_output
        self.radius = arg_radius
        self.raw_shape = arg_shape
        self.probability = arg_probability
        self.pixel_size = arg_pixel_size
        self.logger = logger
        self.logger.info("\U0001F914 starting prediction submodule")

        self.type = "csv"
        self.extensions = EXTENSIONS
        self.abs_input = os.path.abspath(self.raw_input)
        self.model = load_model(
            os.path.abspath(self.fname_model)
        )  # noqa: assignment-from-no-return
        self.logger.info("\U0001F9E0 model imported")

    def __call__(self):
        """Run prediction for all given images."""
        self.logger.info(f"\U0001F4C2 {len(self.file_list)} file(s) found")
        self.logger.info(f"\U0001F5C4 output will be saved to {self.path_output}")

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

    @property
    def shape(self) -> List[str]:
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
        shape_list = shape.split(",")
        return shape_list

    def save_output(self, fname_in: str, df: pd.DataFrame) -> None:
        """Save coordinate list to file with appropriate header."""
        fname_out = os.path.join(self.path_output, f"{basename(fname_in)}.{self.type}")
        df = delete_non_unique_columns(df)
        self.logger.debug(f"non-unique columns to be saved are {df.columns}")

        df.to_csv(fname_out, index=False)
        self.logger.info(
            f"\U0001F3C3 prediction of file {fname_in} saved as {fname_out}"
        )

    def predict_single(
        self, image: np.ndarray, pixel_size: Tuple[float, float]
    ) -> pd.DataFrame:
        """Predict a single (x,y) image accounting for pixel size."""
        column_x = "x [px]" if pixel_size[0] == 1 else "x [µm]"
        column_y = "y [px]" if pixel_size[1] == 1 else "y [µm]"
        columns = [column_y, column_x]  # originally r, c
        if self.probability is not None:
            columns.append("p")

        coords = predict(image, self.model, self.probability)
        df = pd.DataFrame(coords, columns=columns)
        if self.radius is not None:
            df["i"] = get_intensities(image, coords[..., :2], self.radius)

        df[column_x] *= pixel_size[0]
        df[column_y] *= pixel_size[1]
        return df

    def predict_adaptive(self, fname_in: str, image: np.ndarray) -> None:
        """Predict and save a single image."""
        order = ["c", "t", "z", "y", "x"]
        shape = self.shape
        pixel_size = get_pixel_size(self.pixel_size, fname_in, self.logger)

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
                    curr_df = self.predict_single(single_image, pixel_size)
                    curr_df["c"] = c_idx
                    curr_df["t"] = t_idx
                    curr_df["z"] = z_idx
                    df = pd.concat([df, curr_df])

        self.logger.debug(f"completed prediction loop with\n{df.head()}")
        self.save_output(fname_in, df)
