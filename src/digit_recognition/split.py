"""Reproducible train/test partitioning for the scikit-learn Digits dataset."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DigitSplit:
    """Feature arrays, labels, and original sample indices for both partitions."""

    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]


def load_digit_split(*, test_size: float = 0.2, random_state: int = 42) -> DigitSplit:
    """Split digits reproducibly, preserving class proportions.

    Keep the test partition untouched during model selection and tuning.
    Fit preprocessing only on the training partition or within CV pipelines.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer")

    digits = load_digits()
    indices = np.arange(len(digits.target))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=digits.target,
    )
    return DigitSplit(
        X_train=digits.data[train_indices],
        X_test=digits.data[test_indices],
        y_train=digits.target[train_indices],
        y_test=digits.target[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
    )
