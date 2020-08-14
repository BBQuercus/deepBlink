"""Optimizers are used to update weight parameters in a neural network.

The learning rate defines what stepsizes are taken during one iteration of training.
This file contains functions to return standard or custom optimizers.
"""

import tensorflow as tf


def adam(learning_rate: float):
    """Keras' adam optimizer with a specified learning rate."""
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def rmsprop(learning_rate: float):
    """Keras' rmsprop optimizer with a specified learning rate."""
    return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)


def amsgrad(learning_rate: float):
    """Keras' amsgrad optimizer with a specified learning rate."""
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
