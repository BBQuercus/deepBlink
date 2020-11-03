# pylint: skip-file

"""Fully convolutional networks with / without dropout."""

import math

import tensorflow as tf

from ._networks import conv_block
from ._networks import squeeze_block
from ._networks import upconv_block


def conv_squeeze(
    dropout: float = 0.2, cell_size: int = 4, filters: int = 5, n_extra_down: int = 0,
) -> tf.keras.models.Model:
    """Convolutions combined with squeeze network with interspersed dropout.

    Arguments:
        dropout: Percentage of dropout after each convolution+squeeze block.
        cell_size: Size of one cell in the prediction matrix.
        filters: Log2 number of filters in the first convolution+squeeze block.
        n_extra_down: extra downsampling followed by same number of up sampling.
    """
    if not math.log(cell_size, 2).is_integer():
        raise ValueError(f"cell_size must be a power of 2, but is {cell_size}.")

    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    x = inputs
    skip_layers = []

    # Encoder
    for n in range(2 + n_extra_down):
        x = conv_block(inputs=x, filters=2 ** (filters + n), n_convs=3)
        x = squeeze_block(x=x)

        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
        skip_layers.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    skip_bottom = x

    # Decoder
    for n, skip in enumerate(reversed(skip_layers)):
        x = conv_block(inputs=x, filters=2 ** (filters + n), n_convs=3)
        x = squeeze_block(x=x)
        x = upconv_block(inputs=x, skip=skip)

    # # Going back down again
    n_down = int(math.log(cell_size, 2))
    for n in range(n_down):
        x = conv_block(inputs=x, filters=2 ** (filters + n), n_convs=3)
        x = squeeze_block(x=x)
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Concatenate()([skip_bottom, x])

    # Connected
    x = conv_block(inputs=x, filters=2 ** (filters + n), n_convs=3)
    x = squeeze_block(x=x)

    # Logit
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
