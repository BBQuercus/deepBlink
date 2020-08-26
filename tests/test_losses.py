"""Unittests for the deepblink.losses module."""
# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest
import tensorflow as tf

from deepblink.losses import binary_crossentropy
from deepblink.losses import categorical_crossentropy
from deepblink.losses import dice_loss
from deepblink.losses import dice_score
from deepblink.losses import combined_f1_rmse
from deepblink.losses import f1_loss
from deepblink.losses import f1_score
from deepblink.losses import rmse
from deepblink.losses import precision_score
from deepblink.losses import recall_score
from deepblink.losses import combined_bce_rmse


@pytest.fixture
def tensor_true():
    return tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 0.5, 0.5], [1, 0.5, 0.5]]])


@pytest.fixture
def tensor_pred():
    return tf.constant([[[1, 0, 0], [1, 0, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])


def test_binary_crossentropy(tensor_true, tensor_pred):
    assert tf.is_tensor(binary_crossentropy(tensor_true, tensor_pred))


def test_categorical_crossentropy(tensor_true, tensor_pred):
    assert tf.is_tensor(categorical_crossentropy(tensor_true, tensor_pred))


def test_dice_score():
    true_dice = tf.constant([1, 1, 0, 0, 1], dtype=tf.float32)
    pred_dice = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    assert dice_score(true_dice, pred_dice, smooth=0) == tf.constant(
        0.75, dtype=tf.float32
    )


def test_dice_loss(tensor_true, tensor_pred):
    assert tf.is_tensor(dice_loss(tensor_true, tensor_pred))


def test_recall_score(tensor_true, tensor_pred):
    assert recall_score(tensor_true, tensor_pred) == tf.constant(
        2 / 3, dtype=tf.float32
    )


def test_precision_score(tensor_true, tensor_pred):
    assert precision_score(tensor_true, tensor_pred) == tf.constant(
        2 / 3, dtype=tf.float32
    )


def test_f1_score(tensor_true, tensor_pred):
    assert tf.is_tensor(f1_score(tensor_true, tensor_pred))


def test_f1_loss(tensor_true, tensor_pred):
    assert tf.is_tensor(f1_loss(tensor_true, tensor_pred))


def test_rmse():
    true_rmse = tf.constant([[[1, 0, 0], [1, 0.5, 0]], [[0, 0, 0], [1, 0.5, 0.5]]])
    pred_rmse = tf.constant([[[1, 0, 0], [1, 0.5, 0]], [[1, 0.5, 0.5], [0, 0.5, 0.5]]])
    assert rmse(true_rmse, pred_rmse) == tf.constant(0, dtype=tf.float32)


def test_combined_f1_rmse(tensor_true, tensor_pred):
    assert tf.is_tensor(combined_f1_rmse(tensor_true, tensor_pred))


def test_combined_bce_rmse(tensor_true, tensor_pred):
    assert tf.is_tensor(combined_bce_rmse(tensor_true, tensor_pred))
