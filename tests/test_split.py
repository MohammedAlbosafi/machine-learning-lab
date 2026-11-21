"""Regression tests for the held-out evaluation partition."""

import numpy as np
import pytest
from sklearn.datasets import load_digits

from digit_recognition.split import load_digit_split


def test_split_has_expected_shapes_and_disjoint_indices():
    split = load_digit_split()
    assert split.X_train.shape == (1437, 64)
    assert split.X_test.shape == (360, 64)
    assert split.y_train.shape == (1437,)
    assert split.y_test.shape == (360,)
    assert len(np.intersect1d(split.train_indices, split.test_indices)) == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate([split.train_indices, split.test_indices])),
        np.arange(1797),
    )


def test_split_is_reproducible_and_preserves_original_samples():
    first = load_digit_split()
    second = load_digit_split()
    digits = load_digits()
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    np.testing.assert_array_equal(first.X_train, digits.data[first.train_indices])
    np.testing.assert_array_equal(first.y_test, digits.target[first.test_indices])


def test_split_is_stratified():
    split = load_digit_split()
    overall = np.bincount(load_digits().target, minlength=10) / 1797
    train = np.bincount(split.y_train, minlength=10) / len(split.y_train)
    test = np.bincount(split.y_test, minlength=10) / len(split.y_test)
    assert np.max(np.abs(train - overall)) < 0.01
    assert np.max(np.abs(test - overall)) < 0.01


@pytest.mark.parametrize("size", [-0.1, 0, 1, 1.1])
def test_invalid_test_size_is_rejected(size):
    with pytest.raises(ValueError, match="test_size"):
        load_digit_split(test_size=size)


def test_boolean_seed_is_rejected():
    with pytest.raises(TypeError, match="random_state"):
        load_digit_split(random_state=True)
