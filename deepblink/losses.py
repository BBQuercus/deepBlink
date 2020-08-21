"""Functions to calculate training loss on batches of images.

While functions are comparable to metrics, these rely on keras' backend
and do not take raw numpy as input.
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def binary_crossentropy(y_true, y_pred):
    """Keras' binary crossentropy loss."""
    return tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)


def categorical_crossentropy(y_true, y_pred):
    """Keras' categorical crossentropy loss."""
    return tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)


def dice_score(y_true, y_pred, smooth: int = 1):
    r"""Computes the dice coefficient on a batch of tensors.

    .. math::
        \textrm{Dice} = \frac{2 * {\lvert X \cup Y\rvert}}{\lvert X\rvert +\lvert Y\rvert}


    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    Args:
        y_true: Ground truth masks.
        y_pred: Predicted masks.
        smooth: Epslion value to avoid division by zero.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    """Dice score loss corresponding to deepblink.losses.dice_score."""
    return 1 - dice_score(y_true, y_pred)


def recall_score(y_true, y_pred):
    """Recall score metric.

    Defined as ``tp / (tp + fn)`` where tp is the number of true positives and fn the number of false negatives.
    Can be interpreted as the accuracy of finding positive samples or how many relevant samples were selected.
    The best value is 1 and the worst value is 0.
    """
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_pred):
    """Precision score metric.

    Defined as ``tp / (tp + fp)`` where tp is the number of true positives and fp the number of false positives.
    Can be interpreted as the accuracy to not mislabel samples or how many selected items are relevant.
    The best value is 1 and the worst value is 0.
    """
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    r"""F1 score metric.

    .. math::
        F1 = \frac{2 * \textrm{precision} * \textrm{recall}}{\textrm{precision} + \textrm{recall}}

    The equally weighted average of precision and recall.
    The best value is 1 and the worst value is 0.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_value = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_value


def f1_loss(y_true, y_pred):
    """F1 score loss corresponding to deepblink.losses.f1_score."""
    return 1 - f1_score(y_true, y_pred)


def rmse(y_true, y_pred):
    """Calculate root mean square error (rmse) between true and predicted coordinates."""
    coord_true = y_true[..., 1:]
    coord_pred = y_pred[..., 1:]

    comparison = tf.equal(coord_true, tf.constant(0, dtype=tf.float32))

    coord_true_new = tf.where(comparison, tf.zeros_like(coord_true), coord_true)
    coord_pred_new = tf.where(comparison, tf.zeros_like(coord_pred), coord_pred)

    sum_rc_coords = K.sum(coord_true, axis=-1)
    n_true_spots = tf.math.count_nonzero(sum_rc_coords, dtype=tf.float32)

    squared_displacement_xy_summed = K.sum(
        K.square(coord_true_new - coord_pred_new), axis=-1
    )
    rmse_value = K.sqrt(K.sum(squared_displacement_xy_summed) / n_true_spots)

    return rmse_value


def combined_f1_rsme(y_true, y_pred):
    """Difference between F1 score and root mean square error (rmse).

    Optimal value for F1 score is 1 and for rmse is 0.
    Optimal value for the combined score is 1.
    """
    return f1_score(y_true, y_pred) - rmse(y_true, y_pred)
