"""Logging callbacks for wandb."""
# pylint: disable=no-member,missing-function-docstring

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

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


from .data import get_coordinate_list
from .datasets import Dataset
from .metrics import compute_metrics
from .models import Model


def wandb_callback():
    """Callback function to avoid double imports."""
    return wandb.keras.WandbCallback()


class WandbImageLogger(tf.keras.callbacks.Callback):
    """Custom image prediction logger callback in wandb.

    Expects segmentation images and the model class to have a predict_on_image method.

    Attributes:
        model_wrapper: Model used for predictions.
        dataset: Dataset class containing data.
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
            curr_df["image"] = idx  # for downstream groupby's
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
