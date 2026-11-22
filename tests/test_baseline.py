"""Tests for training-only baseline evaluation."""

import pytest

from digit_recognition.baseline import evaluate_baseline


def test_baseline_is_reproducible():
    first = evaluate_baseline()
    second = evaluate_baseline()
    assert first == second
    assert first["training_samples"] == 1437
    assert first["held_out_test_samples"] == 360
    assert len(first["fold_accuracy"]) == 5
    assert 0.09 <= first["accuracy_mean"] <= 0.12
    assert first["balanced_accuracy_mean"] == pytest.approx(0.1)


@pytest.mark.parametrize("folds", [True, 0, 1, 1.5, 1000])
def test_invalid_folds(folds):
    with pytest.raises(ValueError, match="folds"):
        evaluate_baseline(folds=folds)
