"""Functions to calculate training loss on batches of images.

While functions are comparable to metrics, these rely on keras' backend
and do not take raw numpy as input.
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def binary_crossentropy(y_true, y_pred):
    """Return the binary crossentropy loss."""
    return tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)


def categorical_crossentropy(y_true, y_pred):
    """Categorical cross-entropy."""
    return tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)


def dice_coef(y_true, y_pred, smooth: int = 1):
    """Computes the dice coefficient on a batch of tensors.

    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    Args:
        y_true: Ground truth masks.
        y_pred: Predicted masks.
        smooth: Epslion value to avoid division by zero.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """Corresponding dice coefficient loss."""
    return 1 - dice_coef(y_true, y_pred)


def recall_score(y_true, y_pred):
    """Recall score metrics."""
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_pred):
    """Precision score metrics."""
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    """F1 = 2 * (precision*recall) / (precision+recall)."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f1_score_loss(y_true, y_pred):
    """F1 score loss."""
    return 1 - f1_score(y_true, y_pred)


def l2_norm(y_true, y_pred):
    """Calculate L2 norm between true and predicted coordinates."""
    coord_true = y_true[..., 1:]
    coord_pred = y_pred[..., 1:]

    comparison = tf.equal(coord_true, tf.constant(0, dtype=tf.float32))

    coord_true_new = tf.where(comparison, tf.zeros_like(coord_true), coord_true)
    coord_pred_new = tf.where(comparison, tf.zeros_like(coord_pred), coord_pred)

    l2_norm_ = K.sqrt(K.mean(K.sum(K.square(coord_true_new - coord_pred_new), axis=-1)))

    return l2_norm_


def f1_l2_combined_loss(y_true, y_pred):
    """Sum of F1 loss and L2 norm."""
    return l2_norm(y_true, y_pred) + f1_score_loss(y_true, y_pred)
