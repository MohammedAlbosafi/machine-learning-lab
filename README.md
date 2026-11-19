# Handwritten Digit Recognition

This project develops a reproducible multiclass classifier for handwritten digits 0–9, using the scikit-learn Digits dataset. The sprint will establish a baseline, compare candidate models, and evaluate the selected model on a held-out test set.

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

## Project layout

- `src/digit_recognition/` contains the Python package.
- `tests/` checks package installation and bundled dataset availability.
- `pyproject.toml` defines package metadata, dependencies, and pytest discovery.
- `requirements.txt` installs the project in editable mode with its test dependencies.
- [ROADMAP.md](ROADMAP.md) defines the 14-day sprint.
- [LICENSE](LICENSE) contains the MIT license.

Day 1 establishes the package and verifies that the bundled dataset loads with the expected dimensions and labels.
