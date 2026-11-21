import json

from sklearn.datasets import load_digits

from digit_recognition.explore import generate_report, summarize


def test_summary():
    summary = summarize(load_digits())
    assert summary["samples"] == 1797
    assert summary["image_shape"] == [8, 8]
    assert summary["features"] == 64
    assert (summary["pixel_min"], summary["pixel_max"]) == (0, 16)
    assert set(summary["class_counts"]) == set(map(str, range(10)))
    assert sum(summary["class_counts"].values()) == 1797


def test_report(tmp_path):
    output = tmp_path / "report"
    summary = generate_report(output)
    assert json.loads((output / "summary.json").read_text()) == summary
    for name in ("sample_digits.png", "class_distribution.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
