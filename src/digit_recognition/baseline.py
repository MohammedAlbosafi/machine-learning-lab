"""Evaluate a majority-class baseline without touching the held-out test set."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from digit_recognition.split import load_digit_split


def evaluate_baseline(*, folds: int = 5, random_state: int = 42) -> dict:
    """Cross-validate a majority-class predictor on the training partition only."""
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    split = load_digit_split(random_state=random_state)
    if folds > min(np.bincount(split.y_train)):
        raise ValueError("folds cannot exceed the smallest training class count")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        DummyClassifier(strategy="most_frequent"),
        split.X_train,
        split.y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=1,
    )
    return {
        "model": "DummyClassifier(strategy=most_frequent)",
        "training_samples": int(len(split.y_train)),
        "held_out_test_samples": int(len(split.y_test)),
        "folds": folds,
        "random_state": random_state,
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
        "fold_accuracy": [float(x) for x in scores["test_accuracy"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_baseline(folds=args.folds, random_state=args.random_state)
    report = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
