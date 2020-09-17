"""Event handlers for deepblinks CLI."""

from typing import List
import logging
import os
import textwrap
import yaml

import numpy as np
import pandas as pd

from ..inference import get_intensities
from ..inference import predict
from ..io import basename
from ..io import grab_files
from ..io import load_image
from ..io import load_model
from ..io import EXTENSIONS
from ..training import run_experiment
from ..util import delete_non_unique_columns
from ..util import predict_shape
from ..util import train_valid_split


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

        self.abs_output = os.path.abspath(self.raw_output)

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
        self.logger.info("\U0001F686 starting checking submodule")

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
        self.logger.info("\U0001F537 starting checking submodule")

        self.abs_input = os.path.abspath(self.raw_input)

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
        \U000027A1 c: color channels
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
        self.logger.info("\U0001F914 starting prediction submodule")

        self.extensions = EXTENSIONS
        self.abs_input = os.path.abspath(self.raw_input)
        self.model = load_model(
            os.path.abspath(self.fname_model)
        )  # noqa: assignment-from-no-return
        self.logger.info("\U0001F9E0 model imported")

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
        self.logger.debug(f"loading image as RGB {is_rgb}.")
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
                "\U000026A0 images do not have equal shapes (dimensions match)."
            )

        if self.raw_shape is None:
            shape = predict_shape(self.image_list[0].shape)
            self.logger.info(f"\U0001F535 using predicted shape of {shape}.")
        else:
            shape = self.raw_shape
            self.logger.info(f"\U0001F535 using provided input shape of {shape}.")
        for c in ["(", ")", " "]:
            shape = shape.replace(c, "")
        shape = shape.split(",")
        return shape

    def save_output(self, fname_in: str, df: pd.DataFrame) -> None:
        """Save coordinate list to file with appropriate header."""
        fname_out = os.path.join(self.path_output, f"{basename(fname_in)}.{self.type}")
        df = delete_non_unique_columns(df)
        self.logger.debug(f"non-unique columns to be saved are {df.columns}.")

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
            f"\U0001F3C3 prediction of file {fname_in} saved as {fname_out}."
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
            self.logger.debug("axes rearangement did not work properly.")

        # Iterate through c, t, and z
        df = pd.DataFrame()
        for c_idx, t_ser in enumerate(image):
            for t_idx, z_ser in enumerate(t_ser):
                for z_idx, single_image in enumerate(z_ser):
                    curr_df = self.predict_single(single_image, c_idx, t_idx, z_idx)
                    df = df.append(curr_df)

        self.logger.debug(f"completed prediction loop with\n{df.head()}.")
        self.save_output(fname_in, df)

    def run(self):
        """Run prediction for all given images."""
        self.logger.info(f"\U0001F4C2 {len(self.file_list)} file(s) found.")
        self.logger.info(f"\U0001F5C4{' '} output will be saved to {self.path_output}.")

        for fname_in, image in zip(self.file_list, self.image_list):
            self.predict_adaptive(fname_in, image)

        self.logger.info("\U0001F3C1 all predictions are complete.")


class HandleCreate:
    """Handle creation submodule for CLI.

    Args:
        arg_input: Path to folder with images.
        arg_labels: Path to folder with labels.
        arg_name: Name of dataset file to be saved.
        logger: Logger to log verbose output.
    """

    def __init__(self, arg_input, arg_labels, arg_name, logger):
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

    def run(self):
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
