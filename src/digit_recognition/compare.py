"""Compare candidate classifiers using identical training-only validation folds."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from digit_recognition.split import load_digit_split


def compare_models(*, folds: int = 5, random_state: int = 42) -> dict:
    """Return comparable validation scores without scoring the test partition."""
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    split = load_digit_split(random_state=random_state)
    if folds > min(np.bincount(split.y_train)):
        raise ValueError("folds cannot exceed the smallest training class count")
    cv = list(StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state
    ).split(split.X_train, split.y_train))
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state)
        ),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, split.X_train, split.y_train, cv=cv,
            scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1, error_score="raise",
        )
        accuracy = scores["test_accuracy"]
        results[name] = {
            "accuracy_mean": float(accuracy.mean()),
            "accuracy_std": float(accuracy.std()),
            "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
            "fold_accuracy": accuracy.tolist(),
        }
    return {
        "folds": folds,
        "random_state": random_state,
        "training_samples": len(split.y_train),
        "held_out_test_samples": len(split.y_test),
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(compare_models(folds=args.folds, random_state=args.random_state), indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
