"""Inception network architecture."""

import math

import tensorflow as tf

from ._networks import OPTIONS_CONV
from ._networks import conv_block
from ._networks import inception_naive_block
from ._networks import upconv_block
from ._networks import squeeze_block
from ._networks import residual_block
from ._networks import inception_squeeze_block


def inception(
    dropout: float = 0.2,
    cell_size: int = 4,
    filters: int = 6,
    n_extra_down: int = 0,
    spatial: bool = False,
) -> tf.keras.models.Model:
    """Inception combined with squeeze network with interspersed dropout.

    Arguments:
        dropout: Percentage of dropout after each inception block.
        cell_size: Size of one cell in the prediction matrix.
        filters: Log2 number of filters in the first inception block.
        n_extra_down: extra downsampling followed by same number of up sampling.
        spatial: If true, spatial dropout will be used instead of standard dropout.
    """
    if not math.log(cell_size, 2).is_integer():
        raise ValueError(f"cell_size must be a power of 2, but is {cell_size}.")

    inputs = tf.keras.layers.Input(shape=(512, 512, 1))
    # inputs = tf.keras.layers.Input(shape=(None, None, 1))
    x = inputs
    skip_layers = []

    # First conv block
    for _ in range(2):
        x = tf.keras.layers.Conv2D(2 ** filters, **OPTIONS_CONV)(x)
        x = tf.keras.layers.Activation(tf.nn.leaky_relu)(x)
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
        # x = tf.keras.layers.Dropout(dropout)(x)

    # Encoder
    for n in range(2 + n_extra_down):
        x = inception_naive_block(inputs=x, filters=2 ** (filters + n))
        x = squeeze_block(x=x)
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
        # x = tf.keras.layers.Dropout(dropout)(x)
        skip_layers.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Decoder
    for n, skip in enumerate(reversed(skip_layers)):
        x = inception_naive_block(inputs=x, filters=2 ** (filters + 1 - n))
        x = squeeze_block(x=x)
        x = upconv_block(inputs=x, skip=skip, filters=2 ** (filters + 1), n_convs=1)

    # # Going back down again
    n_down = int(math.log(cell_size, 2))
    for n in range(n_down):
        x = inception_naive_block(inputs=x, filters=2 ** (filters + n))
        x = squeeze_block(x=x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Connected
    x = inception_naive_block(inputs=x, filters=2 ** (filters + n_down))
    x = squeeze_block(x=x)

    # Logit
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
