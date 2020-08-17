"""Residual network architecture."""

import tensorflow as tf

from ._networks import conv_block
from ._networks import residual_block
from ._networks import upconv_block


def resnet(dropout: float = 0.2, cell_size: int = 4) -> tf.keras.models.Model:
    """Residual network with interspersed dropout."""
    i = 6  # 64

    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    layers = []

    # Down: halving
    x = conv_block(inputs=inputs, filters=2 ** (i), n_convs=3, dropout=dropout)
    x = residual_block(inputs=x, filters=2 ** (i))
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)
    x = tf.keras.layers.Dropout(dropout)(x)
    layers.append(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Down: halving
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)
    x = residual_block(inputs=x, filters=2 ** (i + 1))
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)
    x = tf.keras.layers.Dropout(dropout)(x)
    layers.append(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Dynamic down sampling
    for n in range(cell_size // 4 - 1):
        x = conv_block(inputs=x, filters=2 ** (i + 1 + n), n_convs=3, dropout=dropout)
        x = residual_block(inputs=x, filters=2 ** (i + 1 + n))
        x = conv_block(inputs=x, filters=2 ** (i + 1 + n), n_convs=3, dropout=dropout)
        x = tf.keras.layers.Dropout(dropout)(x)
        layers.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Dynamic upsampling:
    for n in range(int(2 // cell_size)):
        x = conv_block(inputs=x, filters=2 ** (i + 1 - n), n_convs=3, dropout=dropout)
        x = residual_block(inputs=x, filters=2 ** (i + 1 - n))
        x = conv_block(inputs=x, filters=2 ** (i + 1 - n), n_convs=3, dropout=dropout)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = upconv_block(inputs=x, skip=layers[-(n + 1)], filters=2 ** (i + 1), n_convs=1)

    # Connected
    x = conv_block(inputs=x, filters=2 ** (i + 2), n_convs=3, dropout=dropout)
    x = residual_block(inputs=x, filters=2 ** (i + 2))
    x = conv_block(inputs=x, filters=2 ** (i + 1), n_convs=3, dropout=dropout)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Logit
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
