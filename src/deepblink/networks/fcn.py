"""Fully convolutional networks with / without dropout."""

import tensorflow as tf

from ._networks import OPTIONS_CONV


def fcn() -> tf.keras.models.Model:
    """Simplest FCN architecture without skips or dropout."""
    i = 5  # 32

    inputs = tf.keras.layers.Input(shape=(512, 512, 1))

    # Down: 512 -> 256
    x = tf.keras.layers.Conv2D(filters=2 ** (i), strides=1, **OPTIONS_CONV)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Down: 256 -> 128
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 1), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 1), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Connected
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 3), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 3), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def fcn_dropout() -> tf.keras.models.Model:
    """Simple FCN architecture without skips."""
    i = 6  # 64

    inputs = tf.keras.layers.Input(shape=(512, 512, 1))

    # Down: 512 -> 256
    x = tf.keras.layers.Conv2D(filters=2 ** (i), strides=1, **OPTIONS_CONV)(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Down: 256 -> 128
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 1), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 1), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 1), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Connected
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 3), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=2 ** (i + 3), strides=1, **OPTIONS_CONV)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
