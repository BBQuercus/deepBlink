"""deepBlink — threshold-independent detection and localization of diffraction-limited spots.

Quick start::

    import deepblink

    model  = deepblink.load_model("model.h5")
    image  = deepblink.load_image("image.tif")
    coords = deepblink.predict(image, model, probability=0.5)

Top-level convenience functions (re-exported from submodules):

- :func:`load_model` — load a trained ``.h5`` model from disk.
- :func:`load_image` — load a microscopy image as a numpy array.
- :func:`load_npz` — load a deepBlink dataset (``.npz``).
- :func:`predict` — detect spots in an image using a trained model.
- :func:`get_intensities` — measure integrated intensities around detected spots.
- :func:`compute_metrics` — evaluate predictions against ground truth.
- :func:`train` — run a full training experiment from a config dict.

All submodules remain accessible for advanced usage (e.g.
``deepblink.losses``, ``deepblink.networks``, ``deepblink.augment``).
"""

__version__ = "0.2.0"

from . import augment
from . import cli
from . import data
from . import datasets
from . import inference
from . import io
from . import losses
from . import metrics
from . import models
from . import networks
from . import optimizers
from . import training
from . import util

# ── Core API ──────────────────────────────────────────────────────────
# Loading
from .io import load_model, load_image, load_npz

# Inference
from .inference import predict, predict_tta, get_intensities

# Evaluation
from .metrics import compute_metrics

# Training
from .training import run_experiment as train

__all__ = [
    "load_model",
    "load_image",
    "load_npz",
    "predict",
    "get_intensities",
    "compute_metrics",
    "train",
]
