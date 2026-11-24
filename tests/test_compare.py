import numpy as np
import pytest

from digit_recognition.compare import compare_models
from digit_recognition.baseline import evaluate_baseline


def test_comparison_scores():
    report = compare_models(folds=3)
    assert report["training_samples"] == 1437
    assert report["held_out_test_samples"] == 360
    assert set(report["models"]) == {"majority", "logistic", "svm", "knn"}
    for scores in report["models"].values():
        assert len(scores["fold_accuracy"]) == 3
        assert all(0 <= score <= 1 for score in scores["fold_accuracy"])
        assert scores["accuracy_mean"] == pytest.approx(np.mean(scores["fold_accuracy"]))
        assert 0 <= scores["balanced_accuracy_mean"] <= 1
    assert report["models"]["majority"]["accuracy_mean"] == pytest.approx(
        evaluate_baseline(folds=3)["accuracy_mean"]
    )
    for name in ("logistic", "svm", "knn"):
        assert report["models"][name]["accuracy_mean"] > 0.8


@pytest.mark.parametrize("folds", [True, 1, 2.5, 1000])
def test_invalid_folds(folds):
    with pytest.raises(ValueError):
        compare_models(folds=folds)
