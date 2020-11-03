"""Networks folder.

Contains functions returning the base architectures of used models.
"""

from .conv_squeeze import conv_squeeze
from .inception import inception
from .resnet import resnet


__all__ = [
    "conv_squeeze",
    "inception",
    "resnet",
]
