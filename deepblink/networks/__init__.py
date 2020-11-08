"""Networks folder.

Contains functions returning the base architectures of used models.
"""

from .convolution import convolution
from .inception import inception
from .resnet import resnet


__all__ = [
    "convolution",
    "inception",
    "resnet",
]
