from sklearn.datasets import load_digits


def test_bundled_digits_dataset_is_available():
    digits = load_digits()

    assert digits.images.shape == (1797, 8, 8)
    assert digits.data.shape == (1797, 64)
    assert digits.target.shape == (1797,)
    assert set(digits.target) == set(range(10))
