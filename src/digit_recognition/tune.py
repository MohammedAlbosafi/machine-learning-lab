"""Search candidate hyperparameters without evaluating the held-out partition."""
import argparse
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from digit_recognition.split import load_digit_split


def tune_models() -> dict:
    """Rank candidate configurations by mean validation accuracy."""
    split = load_digit_split(random_state=42)
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
              .split(split.X_train, split.y_train))
    candidates = {
        "logistic": (LogisticRegression(max_iter=2000, random_state=42),
                     {"logisticregression__C": [0.1, 1, 10]}),
        "svm": (SVC(kernel="rbf"),
                {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.001, 0.01]}),
        "knn": (KNeighborsClassifier(),
                {"kneighborsclassifier__n_neighbors": [3, 5, 7, 9],
                 "kneighborsclassifier__weights": ["uniform", "distance"]}),
    }
    results = {}
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(make_pipeline(StandardScaler(), estimator), grid,
                              cv=cv, scoring="accuracy", refit=False,
                              n_jobs=1, error_score="raise")
        search.fit(split.X_train, split.y_train)
        results[name] = {
            "best_params": search.best_params_,
            "best_accuracy": float(search.best_score_),
            "configurations": [
                {"params": params, "mean_accuracy": float(mean),
                 "std_accuracy": float(std)}
                for params, mean, std in zip(search.cv_results_["params"],
                    search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"])
            ],
        }
    return {"folds": 5, "random_state": 42, "models": results,
            "selected_model": max(results, key=lambda name: results[name]["best_accuracy"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(tune_models(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
