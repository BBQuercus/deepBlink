"""Event handlers for deepblinks CLI."""

from typing import List
import logging
import os
import textwrap
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf

from ..inference import get_intensities
from ..inference import predict
from ..io import basename
from ..io import grab_files
from ..io import load_image
from ..io import load_model
from ..io import EXTENSIONS
from ..training import run_experiment
from ..util import delete_non_unique_columns


def predict_shape(shape) -> str:
    """Predict the channel-arangement based on common standards."""
    is_rgb = 3 in shape
    max_len = 5 if is_rgb else 4
    if not any([len(shape) == i for i in range(2, max_len)]):
        raise ValueError("Shape can't be predicted.")

    dims = {}
    dims["x"], dims["y"] = [
        idx for idx, i in enumerate(shape) if i in sorted(shape)[-2:]
    ]
    sorted_shape = sorted(shape)
    if is_rgb:
        dims["3"] = shape.index(3)
        sorted_shape.remove(3)
    if len(sorted_shape) >= 3:
        dims["z"] = shape.index(sorted_shape[0])
    if len(sorted_shape) >= 4:
        dims["t"] = shape.index(sorted_shape[1])

    sorted_dims = [k for k, v in sorted(dims.items(), key=lambda item: item[1])]
    order = ",".join(sorted_dims)
    return order


class HandleConfig:
    """Handle configuration submodule for CLI.

    Args:
        arg_output: Name of output file.
        logger: Logger to log verbose output.
    """

    def __init__(self, arg_output: str, logger: logging.Logger):
        self.raw_output = arg_output
        self.logger = logger
        self.logger.info("\U00002699 starting config submodule")

        self.abs_output = os.path.abspath(arg_output)

    @property
    def config(self):
        """Default configuration as dictionary."""
        return {
            "name": "deepBlink",
            "savedir": "PATH/TO/OUTDIR",
            "comments": "COMMENT ON WANDB",
            "use_wandb": False,
            "dataset": "SpotsDataset",
            "dataset_args": {
                "version": "PATH/TO/DATASET.NPZ",
                "cell_size": 4,
                "flip": False,
                "illuminate": False,
                "rotate": False,
                "gaussian_noise": False,
                "translate": False,
            },
            "model": "SpotsModel",
            "network": "inception_squeeze",
            "network_args": {"dropout": 0.0, "filters": 4, "n_extra_down": 0},
            "loss": "combined_bce_rmse",
            "optimizer": "adam",
            "train_args": {
                "batch_size": 2,
                "epochs": 1000,
                "learning_rate": 1e-4,
                "overfit": False,
            },
        }

    def save_yaml(self):
        """Dump configuration into yaml file."""
        with open(self.abs_output, "w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def run(self):
        """Save configuration as yaml file."""
        self.save_yaml()
        self.logger.info(f"\U0001F3C1 saved config file to {self.abs_output}.")


class HandleTrain:
    """Handle checking submodule for CLI.

    Args:
        arg_config: Path to config.yaml file.
        arg_gpu: Which gpu is to be used.
        logger: Verbose logger.
    """

    def __init__(self, arg_config: str, arg_gpu: int, logger: logging.Logger):
        self.raw_config = arg_config
        self.gpu = arg_gpu

        self.logger = logger

    @property
    def config(self):
        """Load config.yaml file into memory."""
        if not os.path.isfile(self.raw_config):
            raise ImportError(
                "\U0000274C Input file does not exist. Please provide a valid path."
            )
        if not self.raw_config.lower().endswith("yaml"):
            raise ImportError(
                "\U0000274C Input file extension invalid. Please provide the filetype yaml."
            )
        with open(self.raw_config, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.logger.info(f"\U0001F4C2 Loaded config file: {config}")
        return config

    def set_gpu(self):
        """Set GPU environment variable."""
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu}"
        self.logger.info(f"\U0001F5A5 set GPU number to {self.gpu}")

    def run(self):
        """Set configuration and start training loop."""
        self.set_gpu()
        self.logger.info("\U0001F3C3 Beginning with training")
        run_experiment(self.config)
        self.logger.info("\U0001F3C1 training complete")


class HandleCheck:
    """Handle checking submodule for CLI.

    Args:
        arg_input: Path to image.
        logger: Verbose logger.
    """

    def __init__(self, arg_input: str, logger: logging.Logger):
        self.raw_input = arg_input
        self.logger = logger

        self.abs_input = os.path.abspath(arg_input)
        self.logger.info("\U0001F537 starting checking submodule")

    @property
    def image(self):
        """Load a single image."""
        return load_image(self.abs_input)

    def run(self) -> None:
        """Run check for input image."""
        print(
            textwrap.dedent(
                f"""
        1. Your image has a shape of: {self.image.shape}
        ----------
        2. Possible parameters
        \U000027A1 x, y: single 2D image used for one prediction
        \U000027A1 z: third (height) dimension
        \U000027A1 t: time dimension
        \U000027A1 3: RGB color stack
        ----------
        3. By default we would assign: "({predict_shape(self.image.shape)})"
        \U0001F449 If this is incorrect, please provide the proper shape using the --shape flag to the
        submodule predict in deepblink's command line interface
        """
            )
        )


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

        self.extensions = EXTENSIONS
        self.abs_input = os.path.abspath(arg_input)

        self.logger.info("\U0001F914 starting prediction submodule")

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
        self.logger.debug(f"Setting image loaded as RGB {is_rgb}")
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
                "\U000026A0 Images do not have equal shapes (dimensions match)."
            )

        if self.raw_shape is None:
            shape = predict_shape(self.image_list[0].shape)
            self.logger.info(f"\U0001F535 Using predicted shape of {shape}.")
        else:
            shape = self.raw_shape
            self.logger.info(f"\U0001F535 Using provided input shape of {shape}.")
        for c in ["(", ")", " "]:
            shape = shape.replace(c, "")
        shape = shape.split(",")
        return shape

    def save_output(self, fname_in: str, df: pd.DataFrame) -> None:
        """Save coordinate list to file with appropriate header."""
        fname_out = os.path.join(self.path_output, f"{basename(fname_in)}.{self.type}")
        df = delete_non_unique_columns(df)

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

    def predict_adaptive(
        self, fname_in: str, image: np.ndarray, model: tf.keras.models.Model
    ) -> None:
        """Predict and save a single image."""
        order = ["t", "z", "y", "x"]
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
            self.logger.debug("Axes rearangement did not work properly")

        # Iterate through t and z
        df = pd.DataFrame()
        for t_idx, z_ser in enumerate(image):
            for z_idx, single_image in enumerate(z_ser):
                coords = predict(single_image, model)
                curr_df = pd.DataFrame(coords, columns=["r", "c"])
                curr_df["t"] = t_idx
                curr_df["z"] = z_idx
                if self.radius is not None:
                    curr_df["i"] = get_intensities(single_image, coords, self.radius)
                df = df.append(curr_df)

        self.save_output(fname_in, df)

    def run(self):
        """Run prediction for all given images."""
        model = load_model(
            os.path.abspath(self.fname_model)
        )  # noqa: assignment-from-no-return
        self.logger.info("\U0001F9E0 model imported")
        self.logger.info(f"\U0001F4C2 {len(self.file_list)} file(s) found")
        self.logger.info(f"\U0001F5C4{' '} output will be saved to {self.path_output}")

        for fname_in, image in zip(self.file_list, self.image_list):
            self.predict_adaptive(fname_in, image, model)

        self.logger.info("\U0001F3C1 all predictions are complete")
