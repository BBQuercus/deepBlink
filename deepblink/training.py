"""Training functions."""

import os
import platform
import time
from typing import Dict

import matplotlib.pyplot as plt
import tensorflow as tf
import wandb

from .data import get_coordinate_list
from .datasets import Dataset
from .models import Model
from .util import get_from_module


class WandbImageLogger(tf.keras.callbacks.Callback):
    """Custom image prediction logger callback in wandb.

    Expects segmentation images and the model class to have a predict_on_image method.

    Args:
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

    def on_train_begin(self, epochs, logs=None):  # pylint: disable=W0613,W0221
        """Logs the ground truth at train_begin."""
        ground_truth = []
        for i, mask in enumerate(self.train_masks):
            plt.figure()
            plt.imshow(self.train_images[i])
            coord_list = get_coordinate_list(matrix=mask, size_image=self.image_size,)
            plt.scatter(
                coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10
            )
            ground_truth.append(wandb.Image(plt, caption=f"Ground truth train: {i}"))
        wandb.log({"Train ground truth": ground_truth}, commit=False)

        ground_truth_valid = []
        for i, mask in enumerate(self.valid_masks):
            plt.figure()
            plt.imshow(self.valid_images[i])
            coord_list = get_coordinate_list(matrix=mask, size_image=self.image_size,)
            plt.scatter(
                coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10
            )
            ground_truth_valid.append(
                wandb.Image(plt, caption=f"Ground truth valid: {i}")
            )
        wandb.log({"Valid ground truth": ground_truth_valid}, commit=False)

        plt.close(fig="all")

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=W0613
        """Logs predictions on epoch_end."""
        predictions_valid = []
        for i, image in enumerate(self.valid_images):
            plt.figure()
            plt.imshow(image)
            pred_mask = self.model_wrapper.predict_on_image(image)
            coord_list = get_coordinate_list(
                matrix=pred_mask, size_image=self.image_size,
            )
            plt.scatter(
                coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10
            )
            predictions_valid.append(wandb.Image(plt, caption=f"Prediction: {i}"))
        wandb.log({"Predictions valid dataset": predictions_valid}, commit=False)

        predictions_train = []
        for i, image in enumerate(self.train_images):
            plt.figure()
            plt.imshow(image)
            pred_mask = self.model_wrapper.predict_on_image(image)
            coord_list = get_coordinate_list(
                matrix=pred_mask, size_image=self.image_size,
            )
            plt.scatter(
                coord_list[..., 0], coord_list[..., 1], marker="+", color="r", s=10
            )
            predictions_train.append(wandb.Image(plt, caption=f"Prediction: {i}"))
        wandb.log({"Predictions train dataset": predictions_train}, commit=False)

        plt.close(fig="all")


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
        cb_image = WandbImageLogger(model, dataset, cfg["dataset_args"]["cell_size"])
        callbacks.append(cb_image)

        cb_wandb = wandb.keras.WandbCallback()
        callbacks.append(cb_wandb)

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
                    version (str): Name of npz file, e.g. "spots_synt_99659a57"
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
                    *dropout (int): Percentage of dropout only for resnet architecture.
                loss (str): Primary loss, e.g. "binary_crossentropy"
                optimizer (str): Optimizer, e.g. "adam"
                train_args:
                    batch_size (int): Number of images per mini-batch.
                    epochs (int): Total rounds of training.
                    learning_rate (float): Learning rate, e.g. 1e-4
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

    dataset = dataset_class(dataset_args["version"])
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
        wandb.log({"valid_metric": score})
        wandb.join()

    if save_weights:
        model.save_weights()
