import pytest

from digit_recognition.tune import tune_models


@pytest.fixture(scope="module")
def report():
    return tune_models()


def test_search_covers_all_configurations(report):
    assert report["folds"] == 5
    assert report["random_state"] == 42
    assert {name: len(result["configurations"])
            for name, result in report["models"].items()} == {
                "logistic": 3, "svm": 9, "knn": 8}


def test_selection_matches_validation_scores(report):
    models = report["models"]
    for result in models.values():
        scores = result["configurations"]
        assert all(0 <= row["mean_accuracy"] <= 1 for row in scores)
        assert result["best_accuracy"] == max(row["mean_accuracy"] for row in scores)
        assert any(row["params"] == result["best_params"] and
                   row["mean_accuracy"] == result["best_accuracy"] for row in scores)
    assert models[report["selected_model"]]["best_accuracy"] == max(
        result["best_accuracy"] for result in models.values())
