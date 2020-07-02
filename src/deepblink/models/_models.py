"""Model class, to be extended by specific types of models."""

import datetime
import pathlib
from typing import Callable
from typing import Dict
from typing import List

import numpy as np

from ..datasets import Dataset
from ..datasets import SequenceDataset
from ..losses import f1_score
from ..losses import l2_norm

DIRNAME = pathlib.Path(__file__).parents[1].resolve() / "weights"
DATESTRING = datetime.datetime.now().strftime("%Y%d%m_%H%M")


class Model:
    """Base class, to be subclassed by predictors for specific type of data, e.g. spots.

    Args:
        dataset_args: Dataset arguments containing - version, cell_size, flip,
            illuminate, rotate, gaussian_noise, and translate.
        dataset_cls: Specific dataset class.
        network_args: Network arguments containing - n_channels.
        network_fn: Network function returning a built model.
        loss_fn: Loss function.
        optimizer_fn: Optimizer function.
        train_args: Training arguments containing - batch_size, epochs, learning_rate.
        batch_format_fn: Formatting function added in the specific model, e.g. spots.
        batch_augment_fn: Same as batch_format_fn for augmentation.
    """

    def __init__(
        self,
        dataset_args: Dict,
        dataset_cls: Dataset,
        network_args: Dict,
        network_fn: Callable,
        loss_fn: Callable,
        optimizer_fn: Callable,
        train_args: Dict,
        batch_format_fn: Callable = None,
        batch_augment_fn: Callable = None,
    ):
        self.name = f"{DATESTRING}_{self.__class__.__name__}_{dataset_cls.name}_{network_fn.__name__}"

        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

        self.network = network_fn(
            n_channels=network_args["n_channels"]
        )  # **network_args
        self.dataset_args = dataset_args
        self.train_args = train_args
        self.batch_format_fn = batch_format_fn
        self.batch_augment_fn = batch_augment_fn

        try:
            self.load_weights()
        except KeyError:
            print("Training from scratch.")

    @property
    def weights_filename(self) -> str:
        """Return the absolute path to weight file."""
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    @property
    def metrics(self) -> list:
        """Return metrics."""
        return ["accuracy"]

    def fit(
        self, dataset: Dataset, augment_val: bool = True, callbacks: list = None,
    ) -> None:
        """Training loop."""
        if callbacks is None:
            callbacks = []

        self.network.compile(
            loss=self.loss_fn,
            optimizer=self.optimizer_fn(float(self.train_args["learning_rate"])),
            metrics=self.metrics,
        )

        train_sequence = SequenceDataset(
            dataset.x_train,
            dataset.y_train,
            self.train_args["batch_size"],
            format_fn=self.batch_format_fn,
            augment_fn=self.batch_augment_fn,
        )
        valid_sequence = SequenceDataset(
            dataset.x_valid,
            dataset.y_valid,
            self.train_args["batch_size"],
            format_fn=self.batch_format_fn,
            augment_fn=self.batch_augment_fn if augment_val else None,
        )

        self.network.fit(
            train_sequence,
            epochs=self.train_args["epochs"],
            callbacks=callbacks,
            validation_data=valid_sequence,
            shuffle=True,
            # use_multiprocessing=False,
            # workers=1,
        )

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Evaluates on images / masks and return l2 norm and f1 score."""
        if x.ndim < 4:
            x = np.expand_dims(x, -1)

        preds = self.network.predict(x)
        preds = np.float32(preds)
        y_float32 = np.float32(y)

        l2_norm_ = l2_norm(y_float32, preds) * self.dataset_args["cell_size"]
        f1_score_ = f1_score(y_float32, preds)

        return [f1_score_.numpy(), l2_norm_.numpy()]

    def load_weights(self) -> None:
        """Load model weights."""
        self.network.load_weights(self.train_args["pretrained"])

    def save_weights(self) -> None:
        """Save model weights."""
        self.network.save_weights(self.weights_filename)
