"""Network utility functions."""

from typing import Tuple

import tensorflow as tf

# Default options used in all non-logit convolutional layers.
REG = 1e-6
OPTIONS_CONV = {
    "kernel_size": 3,
    "padding": "same",
    "kernel_initializer": "he_normal",
    "kernel_regularizer": tf.keras.regularizers.l2(REG),
    "bias_regularizer": tf.keras.regularizers.l2(REG),
}


def inception_naive_block(
    inputs: tf.keras.layers.Layer, filters: int, efficient: bool = True
) -> tf.keras.layers.Layer:
    """Inception naive block.

    [Conv2d(1,1), Conv2D(3,3), Conv2D(5,5), MaxPooling2D(3,3)] -> output

    Args:
        inputs: Input layer.
        filters: Number of convolutional filters applied.
        efficient: If the. 
    """
    args = {
        "activation": tf.nn.leaky_relu,
        "padding": "same",
        "kernel_initializer": "he_normal",
        # "kernel_regularizer": tf.keras.regularizers.l2(REG),
    }

    # 1x1 conv
    conv1 = tf.keras.layers.Conv2D(filters, (1, 1), **args)(inputs)

    # 3x3 conv
    if efficient:
        conv3 = tf.keras.layers.Conv2D(filters, (1, 1), **args)(inputs)
        conv3 = tf.keras.layers.Conv2D(filters, (3, 3), **args)(conv3)
    else:
        conv3 = tf.keras.layers.Conv2D(filters, (3, 3), **args)(inputs)

    # 5x5 conv
    if efficient:
        conv5 = tf.keras.layers.Conv2D(filters, (1, 1), **args)(inputs)
        conv5 = tf.keras.layers.Conv2D(filters // 2, (5, 5), **args)(conv5)
    else:
        conv5 = tf.keras.layers.Conv2D(filters // 2, (5, 5), **args)(inputs)

    # 3x3 max pool
    pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(inputs)
    if efficient:
        pool = tf.keras.layers.Conv2D(filters // 2, (1, 1), **args)(pool)

    # Concatenate filters
    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def conv_block(
    inputs: tf.keras.layers.Layer, filters: int, n_convs: int = 2, dropout: float = 0
) -> tf.keras.layers.Layer:
    """Convolutional block with optional dropout layer.
    n_convs * (Conv2D -> ReLU -> Optional Dropout).
    Args:
        inputs: Input layer.
        filters: Number of convolutional filters applied.
        n_convs: Number of convolution+relu blocks.
        dropout: If > 0, a dropout layer will be added.
    """
    x = inputs
    for _ in range(n_convs):
        x = tf.keras.layers.Conv2D(filters, **OPTIONS_CONV)(x)
        x = tf.keras.layers.Activation(tf.nn.leaky_relu)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def convpool_block(
    inputs: tf.keras.layers.Layer, filters: int, n_convs: int = 2
) -> tf.keras.layers.Layer:
    """Conv_block with added 2D MaxPooling."""
    x = conv_block(inputs=inputs, filters=filters, n_convs=n_convs)
    x = tf.keras.layers.MaxPooling2D()(x)

    return x


def convpool_skip_block(
    inputs: tf.keras.layers.Layer, filters: int, n_convs: int = 2
) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
    """Conv_block with skip connection.
    Returns:
        skip: Layer to be used as skip connection. Output from conv_block.
        x: Layer to be used in next process. Output from 2D MaxPooling.
    """
    skip = conv_block(inputs=inputs, filters=filters, n_convs=n_convs)
    x = tf.keras.layers.MaxPooling2D()(skip)

    return skip, x


def upconv_block(
    inputs: tf.keras.layers.Layer,
    skip: tf.keras.layers.Layer,
    filters: int,
    n_convs: int = 2,
    dropout: float = 0,
) -> tf.keras.layers.Layer:
    """Upconvolutional block with skip connection concatenation.
    Upsampling -> Conv2D -> ReLU -> Concatenation with skip -> Conv_block.
    Args:
        inputs: Input layer.
        skip: Skip connection input layer.
        filters: Number of convolutional filters applied.
        n_convs: Number of convolution+relu blocks after concatenation.
        dropout: If > 0, a dropout layer will be added.
    """
    x = inputs
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(x)
    # x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Concatenate()([skip, x])
    x = conv_block(inputs=x, filters=filters, n_convs=n_convs, dropout=dropout)

    return x


def residual_block(
    inputs: tf.keras.layers.Layer, filters: int
) -> tf.keras.layers.Layer:
    """Simple residual block with skip connection addition.
    Conv2D -> ReLU (skip) -> Conv2D -> ReLU -> Conv2D -> Addition with skip -> ReLU.
    """
    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(inputs)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu)(x)
    skip = x

    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Conv2D(filters=filters, **OPTIONS_CONV)(x)

    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation(tf.nn.leaky_relu)(x)

    return x


def logit_block(
    inputs: tf.keras.layers.Layer, n_channels: int
) -> tf.keras.layers.Layer:
    """Final decision output with sigmoid/softmax activation depending on n_channels."""
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=1)(inputs)
    if n_channels == 1:
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Activation("softmax")(x)

    return x


# TODO: add input and type checks
def squeeze_block(x: tf.keras.layers.Layer, ratio: int = 8):
    """Squeeze and excitation block.
    ref: https://arxiv.org/pdf/1709.01507.pdf.
    Args:
        x: Input tensor.
        ratio: Number of output filters.
    """
    filters = x.shape[-1]

    x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x1 = tf.keras.layers.Dense(max(filters // ratio, 1))(x1)
    x1 = tf.keras.layers.Activation(tf.nn.leaky_relu)(x1)
    x1 = tf.keras.layers.Dense(filters)(x1)
    x1 = tf.keras.layers.Activation("sigmoid")(x1)
    x = tf.keras.layers.Multiply()([x1, x])
    return x


def inception_squeeze_block(
    inputs: tf.keras.layers.Layer, filters: int, ratio: int = 8
) -> tf.keras.layers.Layer:
    """Inception naive block.
    [Conv2d(1,1), Conv2D(3,3), Conv2D(5,5), MaxPooling2D(3,3)] -> output
    Args:
        inputs: Input layer.
        filters: Number of convolutional filters applied.
    """
    # 1x1 conv
    conv1 = tf.keras.layers.Conv2D(
        filters,
        (1, 1),
        activation=tf.nn.leaky_relu,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(0.0002),
    )(inputs)
    # 3x3 conv
    conv3 = tf.keras.layers.Conv2D(
        filters * 2,
        (3, 3),
        activation=tf.nn.leaky_relu,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(0.0002),
    )(inputs)
    # 5x5 conv
    conv5 = tf.keras.layers.Conv2D(
        filters * 4,
        (5, 5),
        activation=tf.nn.leaky_relu,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(0.0002),
    )(inputs)
    # 3x3 max pooling
    pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(inputs)

    # squeeze before concat
    conv1 = squeeze_block(conv1)
    conv3 = squeeze_block(conv3)
    conv5 = squeeze_block(conv5)
    pool = squeeze_block(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out
