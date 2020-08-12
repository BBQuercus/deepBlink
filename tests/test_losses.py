"""Test losses functions."""

import tensorflow as tf

from deepblink.losses import binary_crossentropy
from deepblink.losses import categorical_crossentropy
from deepblink.losses import dice_score
from deepblink.losses import dice_loss
from deepblink.losses import recall_score
from deepblink.losses import precision_score
from deepblink.losses import f1_score
from deepblink.losses import f1_loss
from deepblink.losses import l2_norm
from deepblink.losses import f1_l2_combined_loss


def test_binary_crossentropy():
    """Test binary crossentropy used in training."""
    true = tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 0.5, 0.5], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert tf.is_tensor(binary_crossentropy(true, pred))


def test_categorical_crossentropy():
    """Test categorical crossentropy used in training."""
    true = tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 0.5, 0.5], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert tf.is_tensor(categorical_crossentropy(true, pred))


def test_dice_score():
    """Test dice score used in training."""
    true = tf.constant([1, 1, 0, 0, 1], dtype=tf.float32)
    pred = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    assert dice_score(true, pred, smooth=0) == tf.constant(0.75, dtype=tf.float32)


def test_dice_loss():
    """Test dice loss used in training."""
    true = tf.constant([1, 1, 0, 0, 1], dtype=tf.float32)
    pred = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    assert tf.is_tensor(dice_loss(true, pred))


def test_recall_score():
    """Test recall used in training."""
    true = tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 0.5, 0.5], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert recall_score(true, pred) == tf.constant(2 / 3, dtype=tf.float32)


def test_precision_score():
    """Test precision used in training."""
    true = tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 0.5, 0.5], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert precision_score(true, pred) == tf.constant(2 / 3, dtype=tf.float32)


def test_f1_score():
    """Test f1 score used in training."""
    true = tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 0.5, 0.5], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert tf.is_tensor(f1_score(true, pred))


def test_f1_loss():
    """Test f1 loss used in training."""
    true = tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 0.5, 0.5], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert tf.is_tensor(f1_loss(true, pred))


def test_l2_norm():
    """Test l2 norm used in training."""
    true = tf.constant([[[1, 0, 0], [1, 0.5, 0]], [[0, 0, 0], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0.5, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert l2_norm(true, pred) == tf.constant(0, dtype=tf.float32)


def test_f1_l2_combined_loss():
    """Test loss that combines f1 score and l2 norm."""
    true = tf.constant([[[1, 0, 0], [1, 0.5, 0]], [[0, 0, 0], [1, 0.5, 0.5]]])
    pred = tf.constant([[[1, 0, 0], [1, 0.5, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert tf.is_tensor(f1_l2_combined_loss(true, pred))
