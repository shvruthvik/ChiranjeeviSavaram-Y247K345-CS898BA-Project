"""
Generates the qualitative preprocessing comparison figure used in the
midterm README under the Data Pipeline and Processing section.

For one sample image from each class, the figure shows:

Raw -> CLAHE -> Gaussian Blur -> Canny Edges
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


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "train",
    "images",
)

OUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "plots",
    "preprocessing_pipeline_comparison.jpg",
)


def main():
    fig, axes = plt.subplots(
        len(CLASSES),
        4,
        figsize=(12, 18),
    )

    stage_names = [
        "Raw",
        "CLAHE",
        "Gaussian Blur",
        "Canny Edges",
    ]

    for row, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls)

        image_files = sorted(
            filename
            for filename in os.listdir(cls_dir)
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        )

        if not image_files:
            raise FileNotFoundError(
                f"No images found in: {cls_dir}"
            )

        sample_file = image_files[0]
        sample_path = os.path.join(
            cls_dir,
            sample_file,
        )

        stages = full_pipeline(sample_path)

        images = [
            stages["raw"],
            stages["clahe"],
            stages["blurred"],
            stages["edges"],
        ]

        for col, (img, stage_name) in enumerate(
            zip(images, stage_names)
        ):
            ax = axes[row, col]

            ax.imshow(
                img,
                cmap="gray",
            )

            ax.axis("off")

            if row == 0:
                ax.set_title(
                    stage_name,
                    fontsize=13,
                    fontweight="bold",
                )

            if col == 0:
                ax.text(
                    -0.15,
                    0.5,
                    cls,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    va="center",
                    ha="right",
                )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(OUT_PATH),
        exist_ok=True,
    )

    plt.savefig(
        OUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()