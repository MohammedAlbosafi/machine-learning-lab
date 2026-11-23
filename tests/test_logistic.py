"""Tests for training-only logistic regression evaluation."""

import pytest

from digit_recognition.logistic import evaluate_logistic


def test_logistic_is_reproducible_and_improves_on_baseline():
    first = evaluate_logistic()
    second = evaluate_logistic()
    assert first == second
    assert first["training_samples"] == 1437
    assert first["held_out_test_samples"] == 360
    assert len(first["fold_accuracy"]) == 5
    assert first["accuracy_mean"] > 0.85
    assert first["balanced_accuracy_mean"] > 0.85


@pytest.mark.parametrize("folds", [True, 0, 1, 1.5, 1000])
def test_invalid_folds(folds):
    with pytest.raises(ValueError, match="folds"):
        evaluate_logistic(folds=folds)
