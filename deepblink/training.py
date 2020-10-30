"""Training functions."""
# pylint: disable=no-member,missing-function-docstring

from typing import Dict
import datetime
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from .data import get_coordinate_list
from .datasets import Dataset
from .metrics import compute_metrics
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
        self, model_wrapper: Model, dataset: Dataset, n_examples: int = 4,
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.valid_images = dataset.x_valid[:n_examples]  # type: ignore[index]
        self.train_images = dataset.x_train[:n_examples]  # type: ignore[index]
        self.train_masks = dataset.y_train[:n_examples]  # type: ignore[index]
        self.valid_masks = dataset.y_valid[:n_examples]  # type: ignore[index]
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

    # pylint: disable=W0613,W0221
    def on_train_begin(self, epochs, logs=None):  # noqa: D102
        self.plot_scatter("Train ground truth", self.train_images, self.train_masks)
        self.plot_scatter("Valid ground truth", self.valid_images, self.valid_masks)

    def on_epoch_end(self, epoch, logs=None):  # noqa: ignore=D102
        self.plot_scatter("Train data predictions", self.train_images)
        self.plot_scatter("Valid data predictions", self.valid_images)

    # pylint: enable=W0613,W0221


class WandbComputeMetrics(tf.keras.callbacks.Callback):
    """Compute the final metrics once training is complete."""

    def __init__(self, model: tf.keras.models.Model, dataset: Dataset, mdist: int):
        super().__init__()
        self.model = model
        self.train_images = dataset.x_train
        self.train_labels = dataset.y_train
        self.valid_images = dataset.x_valid
        self.valid_labels = dataset.y_valid
        self.mdist = mdist

    def log_scores(
        self, name: str, images: np.ndarray, labels: np.ndarray
    ) -> pd.DataFrame:
        """Prediction and logging function for one set of images and labels."""
        df = pd.DataFrame()
        mdist = self.mdist

        for idx, (image, true) in enumerate(zip(images, labels)):
            pred = self.model.predict(image[None, ..., None]).squeeze()
            curr_df = compute_metrics(
                pred=get_coordinate_list(pred, image_size=image.shape[0]),
                true=get_coordinate_list(true, image_size=image.shape[0]),
                mdist=mdist,
            )
            curr_df["image"] = idx
            df = df.append(curr_df)

        # Log single summary values to wandb
        values = {
            f"{name} f1@{mdist} mean": df[df["cutoff"] == mdist]["f1_score"].mean(),
            f"{name} f1@{mdist} std": df[df["cutoff"] == mdist]["f1_score"].std(),
            f"{name} integral mean": df["f1_integral"].mean(),
            f"{name} integral std": df["f1_integral"].std(),
            f"{name} euclidean mean": df["mean_euclidean"].mean(),
            f"{name} euclidean std": df["mean_euclidean"].std(),
        }

        for k, v in values.items():
            wandb.run.summary[k] = v

        # Barplot with all metrics
        try:
            wandb.log(
                {
                    f"{name} metrics": wandb.plot.bar(
                        wandb.Table(
                            data=list(values.items()), columns=["label", "value"]
                        ),
                        "label",
                        "value",
                        title=f"{name} metrics",
                    )
                }
            )
        except TypeError:
            print(list(values.items()))

        return df

    def log_plots(self) -> None:
        """Create matplotlib plots and log to wandb."""
        # F1 score vs. cutoff
        cutoffs = self.df_train["cutoff"].unique()
        plt.errorbar(
            x=cutoffs,
            y=self.df_train.groupby("cutoff")["f1_score"].mean().values,
            yerr=self.df_train.groupby("cutoff")["f1_score"].std().values / 2,
            label="Train",
        )
        plt.errorbar(
            x=cutoffs,
            y=self.df_valid.groupby("cutoff")["f1_score"].mean().values,
            yerr=self.df_valid.groupby("cutoff")["f1_score"].std().values / 2,
            label="Valid",
        )
        plt.legend(loc="lower right")
        wandb.log({"F1 score vs. cutoff": plt})

        # F1 Integral distribution
        plt.hist(x=self.df_train["f1_integral"], label="Train")
        plt.hist(x=self.df_valid["f1_integral"], label="Valid")
        plt.legend(loc="upper left")
        wandb.log({"F1 integral histogram": plt})

    # pylint: disable=W0613
    def on_train_end(self, logs=None):  # noqa: D102
        self.df_train = self.log_scores("Train", self.train_images, self.train_labels)
        self.df_valid = self.log_scores("Valid", self.valid_images, self.valid_labels)
        self.log_plots()

    # pylint: enable=W0613


def train_model(
    model: Model,
    dataset: Dataset,
    cfg: Dict,
    run_name: str = "model",
    use_wandb: bool = True,
) -> Model:
    """Model training loop with callbacks.

    Args:
        model: Model class with the .fit method.
        dataset: Dataset class with access to train and validation images.
        cfg: Configuration file equivalent to the one used in pink.training.run_experiment.
        run_name: Name given to the model.h5 file saved.
        use_wandb: If Wandb should be used.
    """
    callbacks = []

    # def _scheduler(epoch, lr):
    #    if epoch < 100:
    #        return lr
    #    else:
    #        return lr * tf.math.exp(-0.05)

    # cb_schedule = tf.keras.callbacks.LearningRateScheduler(_scheduler)
    # callbacks.append(cb_schedule)

    cb_saver = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(cfg["savedir"], f"{run_name}.h5"), save_best_only=True,
    )
    callbacks.append(cb_saver)

    if use_wandb:
        cb_image = WandbImageLogger(model, dataset)
        cb_wandb = wandb.keras.WandbCallback()
        cb_metrics = WandbComputeMetrics(model, dataset, mdist=3)
        callbacks.extend([cb_image, cb_wandb, cb_metrics])

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
                run_name (str): Name of the current run. Uses format DATE_RUNNAME
                use_wandb (bool): If Wandb should be used.

                savedir (str): Path to where the model should be saved.
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
                    cell_size (int): Size of one cell in the grid, default 4.
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

    network_args = (
        cfg.get("network_args", {}) if cfg.get("network_args", {}) is not None else {}
    )
    dataset_args = cfg.get("dataset_args", {})
    train_args = cfg.get("train_args", {})

    network_args["cell_size"] = dataset_args["cell_size"]

    dataset = dataset_class(dataset_args["version"], dataset_args["cell_size"])

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

    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"{now}_{cfg['run_name']}"

    if use_wandb:
        wandb.init(name=run_name, project=cfg["name"], config=cfg)

    model = train_model(model, dataset, cfg, run_name, use_wandb)

    if use_wandb:
        wandb.join()

    if save_weights:
        model.save_weights()
