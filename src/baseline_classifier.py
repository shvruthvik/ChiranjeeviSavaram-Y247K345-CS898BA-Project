"""
Classical ML baseline for NEU-DET steel surface defect classification.

This script trains a HOG-feature + SVM classifier under two conditions:

    (A) Raw grayscale images

    (B) Preprocessed feature fusion:
        - HOG from CLAHE + Gaussian-blurred images
        - HOG from Canny edge images

The two preprocessed HOG vectors are concatenated so that the classifier
uses both texture and edge information.

Both conditions use the dataset's existing training and validation split
and report Accuracy, Precision, Recall, F1-score, and confusion matrices.

This serves as the traditional machine-learning baseline for comparison
with the project's ResNet50 transfer-learning model.
"""

import os
import sys
import json

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# Allow importing preprocessing.py from the same src folder.
sys.path.append(os.path.dirname(__file__))

from preprocessing import (
    load_grayscale,
    preprocess_for_classifier,
    preprocess_edges_for_classifier,
)


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


# Build paths relative to this Python file.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "train",
    "images",
)

VAL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "validation",
    "images",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "baseline_results.json",
)

TARGET_SIZE = (128, 128)


def list_image_paths(base_dir):
    items = []

    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(base_dir, cls)

        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(".jpg"):
                items.append(
                    (
                        os.path.join(cls_dir, fname),
                        label_idx,
                    )
                )

    return items


def extract_hog_raw(path):
    img = load_grayscale(path)

    img = cv2.resize(
        img,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    return feat


def extract_hog_preprocessed(path):
    """
    Extracts HOG features from two preprocessed representations:

        1. CLAHE-enhanced and Gaussian-blurred image
        2. Canny edge image

    The feature vectors are concatenated to combine texture and
    edge-geometry information.
    """
    blurred = preprocess_for_classifier(
        path,
        target_size=TARGET_SIZE,
    )

    edges = preprocess_edges_for_classifier(
        path,
        target_size=TARGET_SIZE,
    )

    feat_blurred = hog(
        blurred,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    feat_edges = hog(
        edges,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    return np.concatenate(
        [
            feat_blurred,
            feat_edges,
        ]
    )


def build_feature_matrix(items, extractor):
    X = []
    y = []

    for path, label in items:
        X.append(extractor(path))
        y.append(label)

    return np.array(X), np.array(y)


def run_condition(
    name,
    extractor,
    train_items,
    val_items,
):
    print(f"\n=== Condition: {name} ===")

    X_train, y_train = build_feature_matrix(
        train_items,
        extractor,
    )

    X_val, y_val = build_feature_matrix(
        val_items,
        extractor,
    )

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=42,
    )

    clf.fit(
        X_train_s,
        y_train,
    )

    y_pred = clf.predict(X_val_s)

    acc = accuracy_score(
        y_val,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_val,
        y_pred,
    )

    report = classification_report(
        y_val,
        y_pred,
        target_names=CLASSES,
        zero_division=0,
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(report)

    return {
        "condition": name,
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }


def main():
    train_items = list_image_paths(TRAIN_DIR)
    val_items = list_image_paths(VAL_DIR)

    print(
        f"Train images: {len(train_items)} | "
        f"Validation images: {len(val_items)}"
    )

    results = []

    results.append(
        run_condition(
            "Raw -> HOG -> SVM",
            extract_hog_raw,
            train_items,
            val_items,
        )
    )

    results.append(
        run_condition(
            (
                "Preprocessed feature fusion "
                "(CLAHE+Blur+Canny) -> HOG -> SVM"
            ),
            extract_hog_preprocessed,
            train_items,
            val_items,
        )
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()