"""
Generates the qualitative preprocessing comparison figure used in the
midterm README (Data Pipeline & Processing section).

For one sample image per class, shows: Raw -> CLAHE -> Gaussian Blur -> Canny Edges
"""

import os
import matplotlib.pyplot as plt
from preprocessing import full_pipeline

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

DATA_DIR = "../data/train/images"
OUT_PATH = "../outputs/plots/preprocessing_pipeline_comparison.jpg"


def main():
    fig, axes = plt.subplots(len(CLASSES), 4, figsize=(12, 18))
    stage_names = ["Raw", "CLAHE", "Gaussian Blur", "Canny Edges"]

    for row, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls)
        sample_file = sorted(os.listdir(cls_dir))[0]
        sample_path = os.path.join(cls_dir, sample_file)

        stages = full_pipeline(sample_path)
        images = [stages["raw"], stages["clahe"], stages["blurred"], stages["edges"]]

        for col, (img, stage_name) in enumerate(zip(images, stage_names)):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(stage_name, fontsize=13, fontweight="bold")
            if col == 0:
                ax.text(-0.15, 0.5, cls, transform=ax.transAxes,
                         fontsize=11, fontweight="bold",
                         va="center", ha="right", rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
