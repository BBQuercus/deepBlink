"""Regression tests for the deepblink.networks module after Keras 3 migration."""

import numpy as np
import pytest
import keras

from deepblink.networks import unet
from deepblink.networks._networks import (
    conv_block,
    inception_block,
    residual_block,
    squeeze_block,
    upconv_block,
)


@pytest.mark.parametrize("block", ["convolutional", "inception", "residual"])
def test_unet_builds(block):
    """Verify UNet builds and produces correct output shape for each block type."""
    model = unet(cell_size=4, ndown=2, filters=4, block=block)
    assert isinstance(model, keras.Model)

    # Forward pass with a small image
    image = np.random.rand(1, 64, 64, 1).astype(np.float32)
    output = model.predict(image, verbose=0)
    assert output.shape == (1, 16, 16, 3)  # 64 / cell_size=4 = 16


@pytest.mark.parametrize("cell_size", [2, 4, 8])
def test_unet_cell_sizes(cell_size):
    """Verify UNet respects different cell sizes."""
    model = unet(cell_size=cell_size, ndown=2, filters=4)
    image = np.random.rand(1, 64, 64, 1).astype(np.float32)
    output = model.predict(image, verbose=0)
    expected = 64 // cell_size
    assert output.shape == (1, expected, expected, 3)


def test_unet_invalid_cell_size():
    """cell_size must be a power of 2."""
    with pytest.raises(ValueError):
        unet(cell_size=3)


def test_conv_block():
    """Verify conv_block produces correct output."""
    inputs = keras.layers.Input(shape=(32, 32, 1))
    output = conv_block(inputs, filters=16)
    model = keras.Model(inputs=inputs, outputs=output)
    result = model.predict(np.random.rand(1, 32, 32, 1).astype(np.float32), verbose=0)
    assert result.shape == (1, 32, 32, 16)


def test_inception_block():
    """Verify inception_block builds and runs."""
    inputs = keras.layers.Input(shape=(32, 32, 8))
    output = inception_block(inputs, filters=8)
    model = keras.Model(inputs=inputs, outputs=output)
    result = model.predict(np.random.rand(1, 32, 32, 8).astype(np.float32), verbose=0)
    assert result.shape[0] == 1
    assert result.shape[1:3] == (32, 32)
