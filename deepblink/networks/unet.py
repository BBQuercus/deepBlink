"""UNet architecture."""

import math

import tensorflow as tf

from ._networks import OPTIONS_CONV
from ._networks import conv_block
from ._networks import inception_block
from ._networks import residual_block
from ._networks import squeeze_block
from ._networks import upconv_block


def __block(inputs, filters, block, l2):
    opts_conv = OPTIONS_CONV
    opts_conv["kernel_regularizer"] = tf.keras.regularizers.l2(l2) if l2 else None
    opts_conv["bias_regularizer"] = tf.keras.regularizers.l2(l2) if l2 else None
    if block == "convolutional":
        x = conv_block(inputs=inputs, filters=filters, n_convs=3, opts_conv=opts_conv)
    if block == "inception":
        x = inception_block(inputs=inputs, filters=filters, l2_regularizer=l2)
    if block == "residual":
        x = residual_block(inputs=x, filters=filters, opts_conv=opts_conv)
    x = squeeze_block(x=x)
    return x


def __encoder(inputs, filters, block, l2, dropout):
    x = __block(inputs, filters, block, l2)
    skip = tf.keras.layers.SpatialDropout2D(dropout)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(skip)
    return x, skip


def __decoder(inputs, skip, filters, block, l2):
    x = __block(inputs, filters, block, l2)
    x = upconv_block(inputs=x, skip=skip)
    return x


def unet(
    dropout: float = 0.2,
    cell_size: int = 4,
    filters: int = 5,
    ndown: int = 2,
    l2: float = 1e-6,
    block: str = "convolutional",
) -> tf.keras.models.Model:
    """Unet model with second, cell size dependent encoder.

    Note that "convolution" is the currently best block.

    Arguments:
        dropout: Percentage of dropout before each MaxPooling step.
        cell_size: Size of one cell in the prediction matrix.
        filters: Log_2 number of filters in the first inception block.
        ndown: Downsampling steps in the first encoder / decoder.
        l2: L2 value for kernel and bias regularization.
        block: Type of block in each layer. [options: convolutional, inception, residual]
    """
    if not math.log(cell_size, 2).is_integer():
        raise ValueError(f"cell_size must be a power of 2, but is {cell_size}.")

    # Input
    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    x = inputs
    skip_layers = []

    # Encoder v1
    for n in range(ndown):
        x, skip = __encoder(x, 2 ** (filters + n), block, l2, dropout)
        skip_layers.append(skip)
    skip_bottom = x

    # Decoder
    for n, skip in enumerate(reversed(skip_layers)):
        x = __decoder(x, skip, 2 ** (filters + (ndown - n)), block, l2)

    # Encoder v2
    ndown_cell = int(math.log(cell_size, 2))
    for n in range(ndown_cell):
        x, _ = __encoder(x, 2 ** (filters + n), block, l2, dropout)

    # Logit
    if ndown == 2 and cell_size == 4:
        x = tf.keras.layers.Concatenate()([skip_bottom, x])
    x = __block(x, 2 ** (filters + ndown_cell), block, l2)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model
