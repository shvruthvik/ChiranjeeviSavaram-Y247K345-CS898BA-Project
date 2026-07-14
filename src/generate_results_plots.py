"""
Generates quantitative evaluation figures for the baseline classifier.

Outputs:
    - baseline_metric_comparison.jpg
    - confusion_matrices.jpg
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "baseline_results.json",
)

PLOTS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "plots",
)


def main():
    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    raw = results[0]
    prep = results[1]

    generate_metric_comparison(raw, prep)
    generate_confusion_matrices(results)


def generate_metric_comparison(raw, prep):
    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]

    x = np.arange(len(metrics))
    width = 0.35

    raw_values = [
        raw[metric] * 100
        for metric in metrics
    ]

    prep_values = [
        prep[metric] * 100
        for metric in metrics
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    bars_raw = ax.bar(
        x - width / 2,
        raw_values,
        width,
        label="Raw -> HOG -> SVM",
        color="#4C72B0",
    )

    bars_preprocessed = ax.bar(
        x + width / 2,
        prep_values,
        width,
        label=(
            "Preprocessed feature fusion "
            "(CLAHE+Blur+Canny) -> HOG -> SVM"
        ),
        color="#DD8452",
    )

    ax.set_ylabel("Score (%)")

    ax.set_title(
        "Baseline Classifier: Raw vs. Preprocessed Input"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 100)
    ax.legend()

    for bars in [
        bars_raw,
        bars_preprocessed,
    ]:
        for bar in bars:
            height = bar.get_height()

            ax.annotate(
                f"{height:.1f}",
                (
                    bar.get_x()
                    + bar.get_width() / 2,
                    height,
                ),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    metric_plot_path = os.path.join(
        PLOTS_DIR,
        "baseline_metric_comparison.jpg",
    )

    plt.savefig(
        metric_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {metric_plot_path}")


def generate_confusion_matrices(results):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    titles = [
        "Raw -> HOG -> SVM",
        (
            "Preprocessed feature fusion "
            "(CLAHE+Blur+Canny) -> HOG -> SVM"
        ),
    ]

    for ax, result, title in zip(
        axes,
        results,
        titles,
    ):
        confusion = np.array(
            result["confusion_matrix"]
        )

        ax.imshow(
            confusion,
            cmap="Blues",
        )

        ax.set_title(title)

        ax.set_xticks(
            range(len(CLASSES))
        )

        ax.set_yticks(
            range(len(CLASSES))
        )

        ax.set_xticklabels(
            CLASSES,
            rotation=45,
            ha="right",
            fontsize=8,
        )

        ax.set_yticklabels(
            CLASSES,
            fontsize=8,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        threshold = confusion.max() / 2

        for row in range(len(CLASSES)):
            for column in range(len(CLASSES)):
                ax.text(
                    column,
                    row,
                    confusion[row, column],
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if confusion[row, column] > threshold
                        else "black"
                    ),
                    fontsize=9,
                )

    plt.tight_layout()

    confusion_plot_path = os.path.join(
        PLOTS_DIR,
        "confusion_matrices.jpg",
    )

    plt.savefig(
        confusion_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {confusion_plot_path}")


if __name__ == "__main__":
    main()