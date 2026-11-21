"""Summarize the bundled digits data and generate exploratory plots."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits


def summarize(digits):
    labels, counts = np.unique(digits.target, return_counts=True)
    return {
        "samples": len(digits.target),
        "image_shape": list(digits.images.shape[1:]),
        "features": digits.data.shape[1],
        "pixel_min": float(digits.data.min()),
        "pixel_max": float(digits.data.max()),
        "class_counts": {str(label): int(count) for label, count in zip(labels, counts)},
    }


def generate_report(output_dir: Path):
    digits = load_digits()
    summary = summarize(digits)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 5, figsize=(10, 4), layout="constrained")
    for label, ax in zip(range(10), axes.flat):
        index = np.flatnonzero(digits.target == label)[0]
        ax.imshow(digits.images[index], cmap="gray", vmin=0, vmax=16)
        ax.set_title(str(label))
        ax.axis("off")
    fig.savefig(output_dir / "sample_digits.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    counts = summary["class_counts"]
    ax.bar(list(counts), list(counts.values()))
    ax.set(xlabel="Digit", ylabel="Samples", title="Class distribution")
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/digits"))
    args = parser.parse_args()
    print(json.dumps(generate_report(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
