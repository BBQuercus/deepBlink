# pylint: skip-file

"""Fully convolutional networks with / without dropout."""

import math

import tensorflow as tf

from ._networks import conv_block
from ._networks import upconv_block


def fcn(
    dropout: float = 0.2, cell_size: int = 4, filters: int = 6, n_convs: int = 3,
) -> tf.keras.models.Model:
    """Simplest FCN architecture without skips or dropout.

    Arguments:
        dropout: Percentage of dropout only for resnet architecture.
        cell_size: Size of one cell in the prediction matrix.
        filters: Log2 number of filters in the first convolution layers.
        n_convs: Number of convolution layers in each convolution block.
    """
    if not math.log(cell_size, 2).is_integer():
        raise ValueError(f"cell_size must be a power of 2, but is {cell_size}.")

    skip_layers = []
    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    x = inputs

    # Dynamic down sampling: minimum 2 down sampling blocks. Depending on cell_size,
    # add more down sampling blocks to get the correct output shape
    n_down = int(max(2, math.log(cell_size, 2)))
    for n in range(n_down):
        x = conv_block(
            inputs=x, filters=2 ** (filters + n), n_convs=n_convs, dropout=dropout
        )
        skip_layers.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Dynamic upsampling: if cell_size is smaller than 4, after the 2 halvings,
    # we need to upsample int(2 // cell_size) times
    # If cell_size is 1, we need to upsample 2 == int(2 // cell_size)
    # If cell_size is 2, we need to upsample 1 == int(2 // 2)
    n_up = int(2 // cell_size)
    for n in range(n_up):
        x = conv_block(
            inputs=x, filters=2 ** (filters + 1 - n), n_convs=n_convs, dropout=dropout
        )
        x = upconv_block(
            inputs=x, skip=skip_layers[-(n + 1)], filters=2 ** (filters + 1), n_convs=1
        )

    # Connected
    x = conv_block(
        inputs=x, filters=2 ** (filters + 1), n_convs=n_convs, dropout=dropout
    )
    x = conv_block(
        inputs=x, filters=2 ** (filters + 1), n_convs=n_convs, dropout=dropout
    )
    x = tf.keras.layers.Dropout(dropout)(x)

    # Logit
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
