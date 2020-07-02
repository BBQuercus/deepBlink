"""Networks folder.

Contains functions returning the base architectures of used models.
"""

from ._networks import conv_block
from ._networks import convpool_block
from ._networks import convpool_skip_block
from ._networks import logit_block
from ._networks import residual_block
from ._networks import upconv_block
from .fcn import fcn
from .fcn import fcn_dropout
from .resnet import resnet
