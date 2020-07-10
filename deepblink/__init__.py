"""deepBlink for spot detection and localization.

Modules are arranged as follows:
- augment: Data augmentation to artificially increase dataset size.
- cli: Command line interface for inferencing.
- data: Data manipulation. Mainly to properly format for training.
- datasets: Unique data import functions.
- io: File-manipulation-related functions.
- losses: Simple functions returning model losses.
- metrics: Quantitative output of training / model performance.
- models: Training loop containing classes for each type of model.
- networks: Architecture / building of model structure.
- optimizers: Simple functions returning model optimizers.
- util: Basic utility functions not fitting into a category.
"""

__version__ = "0.0.4"

from . import augment
from . import cli
from . import data
from . import datasets
from . import io
from . import losses
from . import metrics
from . import models
from . import networks
from . import optimizers
from . import util
