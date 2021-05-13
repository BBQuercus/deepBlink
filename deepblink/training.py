"""Training functions."""
# pylint: disable=C0415

from typing import Dict
import datetime
import os
import platform

import tensorflow as tf

from .datasets import Dataset
from .models import Model
from .util import get_from_module


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

    cb_saver = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(cfg["savedir"], f"{run_name}.h5"), save_best_only=True,
    )
    callbacks.append(cb_saver)

    if use_wandb:
        from ._wandb import WandbComputeMetrics
        from ._wandb import WandbImageLogger
        from ._wandb import wandb_callback

        cb_image = WandbImageLogger(model, dataset)
        cb_wandb = wandb_callback()
        cb_metrics = WandbComputeMetrics(model, dataset, mdist=3)
        callbacks.extend([cb_image, cb_wandb, cb_metrics])

    model.fit(dataset=dataset, callbacks=callbacks)

    return model


def run_experiment(cfg: Dict, pre_model: tf.keras.models.Model = None):
    """Run a training experiment.

    Configuration file can be generated using deepblink config.

    Args:
        cfg: Dictionary configuration file.
        pre_model: Pre-trained model if not training from scratch.
    """
    # Classes / functions
    dataset_class = get_from_module("deepblink.datasets", cfg["dataset"])
    model_class = get_from_module("deepblink.models", cfg["model"])
    network_fn = get_from_module("deepblink.networks", cfg["network"])
    optimizer_fn = get_from_module("deepblink.optimizers", cfg["optimizer"])
    loss_fn = get_from_module("deepblink.losses", cfg["loss"])

    # Arguments
    augmentation_args = cfg.get("augmentation_args", {})
    dataset_args = cfg.get("dataset_args", {})
    dataset = dataset_class(**dataset_args)
    network_args = (
        cfg.get("network_args", {}) if cfg.get("network_args", {}) is not None else {}
    )
    network_args["cell_size"] = dataset_args["cell_size"]
    train_args = cfg.get("train_args", {})

    model = model_class(
        augmentation_args=augmentation_args,
        dataset_args=dataset_args,
        dataset_cls=dataset,
        loss_fn=loss_fn,
        network_args=network_args,
        network_fn=network_fn,
        optimizer_fn=optimizer_fn,
        train_args=train_args,
        pre_model=pre_model,
    )

    cfg["system"] = {
        "gpus": tf.config.list_logical_devices("GPU"),
        "version": platform.version(),
        "platform": platform.platform(),
    }

    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"{now}_{cfg['run_name']}"

    use_wandb = cfg["use_wandb"]
    if use_wandb:
        try:
            import wandb

            if wandb.__version__ <= "0.10.03":
                raise AssertionError
        except (ModuleNotFoundError, AttributeError, AssertionError):
            raise ImportError(
                (
                    "To support conda packages we don't ship deepBlink with wandb. "
                    "Please install any using pip: 'pip install \"wandb>=0.10.3\"'"
                )
            )

        # pylint:disable=E1101
        wandb.init(name=run_name, project=cfg["name"], config=cfg)

    model = train_model(model, dataset, cfg, run_name, use_wandb)

    if use_wandb:
        wandb.join()  # pylint:disable=E1101
