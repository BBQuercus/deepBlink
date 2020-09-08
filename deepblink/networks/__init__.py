"""Networks folder.

Contains functions returning the base architectures of used models.
"""

from ._networks import conv_block
from ._networks import convpool_block
from ._networks import convpool_skip_block
from ._networks import inception_naive_block
from ._networks import logit_block
from ._networks import residual_block
from ._networks import upconv_block
from .fcn import fcn
from .inception import conv_inception
from .inception import inception
from .resnet import resnet


__all__ = [
    "conv_block",
    "convpool_block",
    "convpool_skip_block",
    "conv_inception",
    "inception_naive_block",
    "logit_block",
    "residual_block",
    "upconv_block",
    "fcn",
    "inception",
    "resnet",
]
