"""Network utility functions."""

from typing import Tuple

import tensorflow as tf

OPTIONS_CONV = {"kernel_size": 3, "padding": "same", "kernel_initializer": "he_normal"}


def conv_block(
    inputs: tf.keras.layers.Layer, filters: int, n_convs: int = 2, dropout: float = 0
) -> tf.keras.layers.Layer:
    """Convolutional block."""
    x = inputs
    for _ in range(n_convs):
        x = tf.keras.layers.Conv2D(filters, **OPTIONS_CONV)(x)
        x = tf.keras.layers.Activation("relu")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    return x


def convpool_block(
    inputs: tf.keras.layers.Layer, filters: int, n_convs: int = 2
) -> tf.keras.layers.Layer:
    """n_convs * (Convolution -> ReLU) -> MaxPooling."""
    x = conv_block(inputs=inputs, filters=filters, n_convs=n_convs)
    x = tf.keras.layers.MaxPooling2D()(x)

    return x


def convpool_skip_block(
    inputs: tf.keras.layers.Layer, filters: int, n_convs: int = 2
) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
    """n_convs * (Convolution -> ReLU) -> MaxPooling."""
    skip = conv_block(inputs=inputs, filters=filters, n_convs=n_convs)
    x = tf.keras.layers.MaxPooling2D()(skip)

    return skip, x


def upconv_block(
    inputs: tf.keras.layers.Layer,
    skip: tf.keras.layers.Layer,
    filters: int,
    n_convs: int = 2,
) -> tf.keras.layers.Layer:
    """Upsampling -> Conv -> Concat -> Conv."""
    x = inputs
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = conv_block(inputs=x, filters=filters, n_convs=n_convs)

    return x


def residual_block(
    inputs: tf.keras.layers.Layer, filters: int
) -> tf.keras.layers.Layer:
    """Simple residual block with addition of skips."""
    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    skip = x

    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)

    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation("relu")(x)

    return x


def logit_block(
    inputs: tf.keras.layers.Layer, n_channels: int
) -> tf.keras.layers.Layer:
    """Final decision output."""
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1)(inputs)
    if n_channels == 1:
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Activation("softmax")(x)

    return x
