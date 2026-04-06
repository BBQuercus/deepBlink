"""Network utility functions."""

import keras

# Default options used in all non-logit convolutional layers.
REG = 1e-6
OPTIONS_CONV = {
    "kernel_size": 3,
    "padding": "same",
    "kernel_initializer": "he_normal",
    "kernel_regularizer": keras.regularizers.l2(REG),
    "bias_regularizer": keras.regularizers.l2(REG),
}


def inception_block(
    inputs: keras.layers.Layer,
    filters: int,
    efficient: bool = True,
    l2_regularizer: float = 1e-6,
) -> keras.layers.Layer:
    """Inception block.

    Args:
        inputs: Input layer.
        filters: Number of convolutional filters applied.
        efficient: If defined, use a more efficient inception block.
        l2_regularizer: L2 value for kernel and bias regularization.
    """
    args = {
        "activation": keras.activations.leaky_relu,
        "padding": "same",
        "kernel_initializer": "he_normal",
        "kernel_regularizer": keras.regularizers.l2(l2_regularizer),
        "bias_regularizer": keras.regularizers.l2(l2_regularizer),
    }

    # 1x1 conv
    conv1 = keras.layers.Conv2D(filters, (1, 1), **args)(inputs)

    # 3x3 conv
    if efficient:
        conv3 = keras.layers.Conv2D(filters, (1, 1), **args)(inputs)
        conv3 = keras.layers.Conv2D(filters, (3, 3), **args)(conv3)
    else:
        conv3 = keras.layers.Conv2D(filters, (3, 3), **args)(inputs)

    # 5x5 conv
    if efficient:
        conv5 = keras.layers.Conv2D(filters, (1, 1), **args)(inputs)
        conv5 = keras.layers.Conv2D(filters // 2, (5, 5), **args)(conv5)
    else:
        conv5 = keras.layers.Conv2D(filters // 2, (5, 5), **args)(inputs)

    # 3x3 max pool
    pool = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(inputs)
    if efficient:
        pool = keras.layers.Conv2D(filters // 2, (1, 1), **args)(pool)

    # Concatenate filters
    layer_out = keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    n_convs: int = 2,
    opts_conv: dict = OPTIONS_CONV,
) -> keras.layers.Layer:
    """Convolutional block with optional dropout layer.

    Args:
        inputs: Input layer.
        filters: Number of convolutional filters applied.
        n_convs: Number of convolution+relu blocks.
        opts_conv: Options passed to Conv2D.
    """
    x = inputs
    for _ in range(n_convs):
        x = keras.layers.Conv2D(filters, **opts_conv)(x)
        x = keras.layers.Activation(keras.activations.leaky_relu)(x)
    return x


def upconv_block(
    inputs: keras.layers.Layer, skip: keras.layers.Layer
) -> keras.layers.Layer:
    """Upconvolutional block with skip connection concatenation.

    Args:
        inputs: Input layer.
        skip: Skip connection input layer.
    """
    x = inputs
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Concatenate()([skip, x])

    return x


def residual_block(
    inputs: keras.layers.Layer, filters: int, opts_conv: dict = OPTIONS_CONV
) -> keras.layers.Layer:
    """Simple residual block."""
    x = keras.layers.Conv2D(filters=filters, **opts_conv)(inputs)
    x = keras.layers.Activation(keras.activations.leaky_relu)(x)
    skip = x

    x = keras.layers.Conv2D(filters=filters, **opts_conv)(x)
    x = keras.layers.Activation(keras.activations.leaky_relu)(x)
    x = keras.layers.Conv2D(filters=filters, **opts_conv)(x)

    x = keras.layers.Add()([x, skip])
    x = keras.layers.Activation(keras.activations.leaky_relu)(x)

    return x


# TODO: add input and type checks
def squeeze_block(x: keras.layers.Layer, ratio: int = 8):
    """Squeeze and excitation block.

    ref: https://arxiv.org/pdf/1709.01507.pdf.

    Args:
        x: Input tensor.
        ratio: Number of output filters.
    """
    filters = x.shape[-1]

    x1 = keras.layers.GlobalAveragePooling2D()(x)
    x1 = keras.layers.Dense(max(filters // ratio, 1))(x1)
    x1 = keras.layers.Activation(keras.activations.leaky_relu)(x1)
    x1 = keras.layers.Dense(filters)(x1)
    x1 = keras.layers.Activation("sigmoid")(x1)
    x = keras.layers.Multiply()([x1, x])
    return x
