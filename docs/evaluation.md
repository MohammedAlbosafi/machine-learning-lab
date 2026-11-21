# Evaluation protocol

Use `load_digit_split()` to reserve a stratified 20% test partition with a fixed random seed. The default split contains 1,437 training examples and 360 held-out examples.

```python
from digit_recognition.split import load_digit_split

split = load_digit_split()
print(split.X_train.shape, split.X_test.shape)
```

Use the training partition for cross-validation, hyperparameter selection, and preprocessing decisions. Fit transformations within a training pipeline to prevent leakage between cross-validation folds. Evaluate the selected model on the held-out test partition only after model selection is complete.

The returned original sample indices make it possible to audit partition membership and reproduce results. The Digits dataset contains 8 × 8 grayscale samples from scikit-learn; no external data is downloaded.
