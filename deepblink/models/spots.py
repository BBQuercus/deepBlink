"""SpotsModel class."""

import functools

import numpy as np

from ..augment import augment_batch_baseline
from ..losses import combined_f1_rmse
from ..losses import f1_score
from ..losses import rmse
from ._models import Model


class SpotsModel(Model):
    """Class to predict spot localization; see base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.batch_augment_fn = functools.partial(
            augment_batch_baseline,
            flip_=self.augmentation_args["flip"],
            illuminate_=self.augmentation_args["illuminate"],
            gaussian_noise_=self.augmentation_args["gaussian_noise"],
            rotate_=self.augmentation_args["rotate"],
            translate_=self.augmentation_args["translate"],
            cell_size=self.dataset_args["cell_size"],
        )

    @property
    def metrics(self) -> list:
        """List of all metrics recorded during training."""
        return [
            f1_score,
            rmse,
            combined_f1_rmse,
        ]

    def predict_on_image(self, image: np.ndarray) -> np.ndarray:
        """Predict on a single input image."""
        return self.network.predict(image[None, ..., None], batch_size=1).squeeze()
