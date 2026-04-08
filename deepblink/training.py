"""Training functions."""

import datetime
import math
import os
import platform

import keras
import numpy as np
import tensorflow as tf

from .datasets import Dataset
from .models import Model
from .util import get_from_module


def _build_experiment_callbacks(cfg: dict) -> list[keras.callbacks.Callback]:
    """Build additional callbacks from experiment config keys in train_args.

    Reads optional keys: ``early_stopping``, ``lr_schedule``, ``mixed_precision``.
    """
    ta = cfg.get("train_args", {})
    callbacks: list[keras.callbacks.Callback] = []

    # Early stopping
    if ta.get("early_stopping", False):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                patience=ta.get("early_stopping_patience", 20),
                restore_best_weights=True,
            )
        )

    # LR schedules
    schedule = ta.get("lr_schedule", None)
    epochs = ta.get("epochs", 200)
    lr = float(ta.get("learning_rate", 1e-4))

    if schedule == "cosine":
        lr_min = float(ta.get("lr_min", 1e-6))
        callbacks.append(
            keras.callbacks.LearningRateScheduler(
                lambda epoch: lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * epoch / epochs))
            )
        )
    elif schedule == "plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        )
    elif schedule == "warmup_cosine":
        warmup_epochs = ta.get("warmup_epochs", 10)
        lr_min = float(ta.get("lr_min", 1e-6))

        def _warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return lr * (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))

        callbacks.append(keras.callbacks.LearningRateScheduler(_warmup_cosine))

    # Mixed precision
    if ta.get("mixed_precision", False):
        keras.mixed_precision.set_global_policy("mixed_float16")

    return callbacks


def train_model(
    model: Model,
    dataset: Dataset,
    cfg: dict,
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

    cb_saver = keras.callbacks.ModelCheckpoint(
        os.path.join(cfg["savedir"], f"{run_name}.h5"), save_best_only=True,
    )
    callbacks.append(cb_saver)

    # Experiment callbacks (LR schedules, early stopping, etc.)
    callbacks.extend(_build_experiment_callbacks(cfg))

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


def run_experiment(cfg: dict, pre_model: keras.Model = None):
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

        wandb.init(name=run_name, project=cfg["name"], config=cfg)

    model = train_model(model, dataset, cfg, run_name, use_wandb)

    if use_wandb:
        wandb.join()
