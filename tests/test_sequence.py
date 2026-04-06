"""Regression tests for SequenceDataset after Keras 3 PyDataset migration."""

import numpy as np
import pytest

from deepblink.datasets.sequence import SequenceDataset


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    x = np.random.rand(20, 32, 32).astype(np.float32)
    y = np.random.rand(20, 8, 8, 3).astype(np.float32)
    return x, y


def test_sequence_length(sample_data):
    """Verify __len__ returns correct batch count."""
    x, y = sample_data
    seq = SequenceDataset(x, y, batch_size=4)
    assert len(seq) == 5  # 20 / 4


def test_sequence_getitem(sample_data):
    """Verify __getitem__ returns properly shaped batches."""
    x, y = sample_data
    seq = SequenceDataset(x, y, batch_size=4)
    batch_x, batch_y = seq[0]
    assert batch_x.shape == (4, 32, 32, 1)  # expanded dim
    assert batch_y.shape == (4, 8, 8, 3)


def test_sequence_epoch_end(sample_data):
    """Verify on_epoch_end shuffles data."""
    x, y = sample_data
    seq = SequenceDataset(x, y, batch_size=4)
    original_first = seq.x[0].copy()
    # Run many times to ensure at least one shuffle differs
    shuffled = False
    for _ in range(10):
        seq.on_epoch_end()
        if not np.array_equal(seq.x[0], original_first):
            shuffled = True
            break
    assert shuffled


def test_sequence_overfit(sample_data):
    """Verify overfit mode always returns first batch."""
    x, y = sample_data
    seq = SequenceDataset(x, y, batch_size=4, overfit=True)
    batch_0 = seq[0]
    batch_1 = seq[1]
    np.testing.assert_array_equal(batch_0[0], batch_1[0])


def test_sequence_batch_larger_than_data():
    """Verify warning when batch_size > dataset size."""
    x = np.random.rand(3, 8, 8).astype(np.float32)
    y = np.random.rand(3, 2, 2, 3).astype(np.float32)
    seq = SequenceDataset(x, y, batch_size=10)
    with pytest.warns(RuntimeWarning):
        length = len(seq)
    assert length == 1
