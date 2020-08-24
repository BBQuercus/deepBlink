"""Training functions."""

from typing import Dict
import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb

from .data import get_coordinate_list
from .datasets import Dataset
from .models import Model
from .util import get_from_module


class WandbImageLogger(tf.keras.callbacks.Callback):
    """Custom image prediction logger callback in wandb.

    Expects segmentation images and the model class to have a predict_on_image method.

    Attributes:
        model_wrapper: Model used for predictions.
        dataset: Dataset class containing data.
        cell_size: Size of one cell in the grid.
        n_examples: Number of examples saved for display.
    """

    def __init__(
        self,
        model_wrapper: Model,
        dataset: Dataset,
        cell_size: int = 4,
        n_examples: int = 4,
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.valid_images = dataset.x_valid[:n_examples]  # type: ignore[index]
        self.train_images = dataset.x_train[:n_examples]  # type: ignore[index]
        self.train_masks = dataset.y_train[:n_examples]  # type: ignore[index]
        self.valid_masks = dataset.y_valid[:n_examples]  # type: ignore[index]
        self.cell_size = cell_size
        self.image_size = dataset.x_train[0].shape[0]  # type: ignore[index]

    def plot_scatter(
        self, title: str, images: np.ndarray, masks: np.ndarray = None
    ) -> None:
        """Plot one set of images to wandb."""
        plots = []
        for i, image in enumerate(images):
            if masks is not None:
                mask = masks[i]
            else:
                mask = self.model_wrapper.predict_on_image(image)  # type: ignore[attr-defined]
            coords = get_coordinate_list(mask, image_size=self.image_size)

            plt.figure()
            plt.imshow(image)
            plt.scatter(coords[..., 1], coords[..., 0], marker="+", color="r", s=10)
            plots.append(wandb.Image(plt, caption=f"{title}: {i}"))
        wandb.log({title: plots}, commit=False)
        plt.close(fig="all")

    def on_train_begin(self, epochs, logs=None):  # pylint: disable=W0613,W0221
        """Logs the ground truth at train_begin."""
        self.plot_scatter("Train ground truth", self.train_images, self.train_masks)
        self.plot_scatter("Valid ground truth", self.valid_images, self.valid_masks)

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=W0613
        """Logs predictions on epoch_end."""
        self.plot_scatter("Train data predictions", self.train_images)
        self.plot_scatter("Valid data predictions", self.valid_images)


def train_model(
    model: Model, dataset: Dataset, cfg: Dict, use_wandb: bool = True
) -> Model:
    """Model training loop with callbacks."""
    callbacks = []

    cb_saver = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(cfg["savedir"], f"{cfg['name']}_{int(time.time())}.h5"),
        save_best_only=True,
    )
    callbacks.append(cb_saver)

    if use_wandb:
        cb_image = WandbImageLogger(
            model, dataset, cell_size=cfg["dataset_args"]["cell_size"]
        )
        cb_wandb = wandb.keras.WandbCallback()
        callbacks.extend([cb_image, cb_wandb])

    model.fit(dataset=dataset, callbacks=callbacks)

    return model


def run_experiment(cfg: Dict, save_weights: bool = False):
    """Run a training experiment.

    An example of the configuration file below can be viewed in the bin/ directory of the github repository.
    NOTE - There are currently only one type of dataset and model option. This is intentional to make
    future development easier of new models such as 3D / 4D options.

    Args:
        cfg: Dictionary configuration file.
            Usually through a parsed yaml file (example in bin/) in the following format: ::

                name (str): Name of the Wandb project.
                comments (str): Comments on runs.
                savedir (str): Path to where the model should be saved.
                use_wandb (bool): If Wandb should be used.
                dataset (str): Name of dataset class, e.g. "SpotsDataset"
                dataset_args:
                    version (str): Path to dataset.npz file.
                    cell_size (int): Size of one cell in the grid.
                    flip (bool): If flipping should be used as augmentation.
                    illuminate (bool): If illumination should be used as augmentation.
                    rotate (bool): If rotation should be used as augmentation.
                    gaussian_noise (bool): If gaussian noise should be added as augmentation.
                    translate (bool): If translation should be used as augmentation.
                model (str): Name of the model class, e.g. "SpotsModel"
                network (str): Name of the network architecture, e.g. "resnet"
                network_args:
                    Arguments passed to the network function.
                    dropout (float): Percentage of dropout only for resnet architecture, default 0.2.
                    cell_size (int): Size of one cell in the grid, default 4.
                    filters (int): log2 number of filters in the first convolution layers, default 6.
                    n_convs (int): number of convolution layers in each convolution block, default 3.
                    conv_after_res: If True, adds additional convolution block after residual block, default True.
                loss (str): Primary loss, e.g. "binary_crossentropy"
                optimizer (str): Optimizer, e.g. "adam"
                train_args:
                    batch_size (int): Number of images per mini-batch.
                    epochs (int): Total rounds of training.
                    learning_rate (float): Learning rate, e.g. 1e-4
                    overfit (bool): If model should overfit to one batch.
                    pretrained (str): Optional weights file to jumpstart training.


        save_weights: If model weights should be saved separately.
            The complete model is automatically saved.
    """
    dataset_class = get_from_module("deepblink.datasets", cfg["dataset"])
    model_class = get_from_module("deepblink.models", cfg["model"])
    network_fn = get_from_module("deepblink.networks", cfg["network"])
    optimizer_fn = get_from_module("deepblink.optimizers", cfg["optimizer"])
    loss_fn = get_from_module("deepblink.losses", cfg["loss"])

    network_args = cfg.get("network_args", {})
    dataset_args = cfg.get("dataset_args", {})
    train_args = cfg.get("train_args", {})

    network_args["cell_size"] = dataset_args["cell_size"]

    dataset = dataset_class(dataset_args["version"], dataset_args["cell_size"])
    dataset.load_data()

    use_wandb = cfg["use_wandb"]
    model = model_class(
        dataset_args=dataset_args,
        dataset_cls=dataset,
        loss_fn=loss_fn,
        network_args=network_args,
        network_fn=network_fn,
        optimizer_fn=optimizer_fn,
        train_args=train_args,
    )

    cfg["system"] = {
        "gpus": tf.config.list_logical_devices("GPU"),
        "version": platform.version(),
        "platform": platform.platform(),
    }

    if use_wandb:
        wandb.init(project=cfg["name"], config=cfg)

    model = train_model(model, dataset, cfg, use_wandb)

    if use_wandb:
        score = model.evaluate(dataset.x_valid, dataset.y_valid)
        wandb.log({"valid_metric": score[0]})
        wandb.join()

    if save_weights:
        model.save_weights()
