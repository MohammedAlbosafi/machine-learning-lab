# Handwritten Digit Recognition

This project develops a reproducible multiclass classifier for handwritten digits 0–9, using the scikit-learn Digits dataset. The goal is to establish a baseline, compare candidate models, and evaluate the selected model on a held-out test set.

## Dataset

The [Digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) contains 1,797 grayscale images of size 8 × 8, with pixel values from 0 to 16. Each image is represented by 64 features and paired with its digit label. The dataset ships with scikit-learn and loads locally through `sklearn.datasets.load_digits`.

## Development setup

Use Python 3.11 or newer. From the repository root, create a virtual environment and install the package and test dependencies.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

In PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`.

## Dataset exploration

Generate the dataset summary and plots locally:

```sh
python -m digit_recognition.explore --output-dir reports/digits
```

The command writes `summary.json`, `sample_digits.png` and `class_distribution.png`. The dataset contains 174–183 samples per class, with pixel values ranging from 0 to 16. The classes are roughly balanced.

## Majority-class baseline

Run a reproducible five-fold stratified cross-validation baseline on the training partition:

```sh
python -m digit_recognition.baseline --output reports/digits/baseline.json
```

The baseline always predicts the most frequent class in each training fold. It provides a reference point for future classifiers, not a useful recognition model. On the default split and seed, mean cross-validation accuracy is approximately 10.09% and balanced accuracy is 10.00%. The held-out test partition is not evaluated during this step.

## Candidate model comparison

Compare the majority-class baseline, logistic regression, RBF support vector machine and five-neighbor classifier using identical stratified validation folds:

```sh
python -m digit_recognition.compare --output reports/digits/comparison.json
```

The report includes mean accuracy, fold-level accuracy, standard deviation and mean balanced accuracy. Scaling is fitted within each training fold for the three candidate models. The held-out test partition is reserved for final evaluation.

## Hyperparameter search

Search 20 candidate configurations using five identical stratified validation folds:

```sh
python -m digit_recognition.tune --output reports/digits/tuning.json
```

The report records every configuration's mean and standard deviation of validation accuracy, the best settings per model, and the selected model. Scaling is fitted within each fold. The held-out test set is not scored, and the selected estimator is not refitted during the search.

## Project layout

- `src/digit_recognition/` contains the Python package.
- `tests/` checks package installation and bundled dataset availability.
- `pyproject.toml` defines package metadata, dependencies, and pytest discovery.
- `requirements.txt` installs the project in editable mode with its test dependencies.
- [ROADMAP.md](ROADMAP.md) outlines the planned experiments and features.
- [LICENSE](LICENSE) contains the MIT license.

Dataset exploration, a reproducible held-out partition, and a training-only majority-class baseline are available. Candidate models can be compared using the same training-only protocol.
