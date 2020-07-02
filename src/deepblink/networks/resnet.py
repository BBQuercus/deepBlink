"""Residual network architecture."""

import tensorflow as tf

from ._networks import conv_block
from ._networks import residual_block


def resnet(n_channels: int = 3, dropout: float = 0.2) -> tf.keras.models.Model:
    """Residual network with interspersed dropout."""
    i = 6  # 64

    inputs = tf.keras.layers.Input(shape=(512, 512, 1))

    # Down: 512 -> 256
    x = conv_block(inputs=inputs, filters=2 ** (i), n_convs=3, dropout=dropout)
    x = residual_block(inputs=x, filters=2 ** (i))
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)

    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Down: 256 -> 128
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)
    x = residual_block(inputs=x, filters=2 ** (i + 1))
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)

    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Connected
    x = conv_block(inputs=x, filters=2 ** (i + 2), n_convs=3, dropout=dropout)
    x = residual_block(inputs=x, filters=2 ** (i + 2))
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)

    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
