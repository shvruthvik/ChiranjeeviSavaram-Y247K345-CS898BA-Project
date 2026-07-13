"""
Classical ML baseline for NEU-DET steel surface defect classification.

Rationale (see README "Baseline Implementation"):
This sandbox has no GPU / PyTorch / TensorFlow / internet access, so the
planned ResNet50 transfer-learning model (see src/resnet50_transfer.py)
cannot be executed here. As a substitute -- and as a legitimate baseline
in its own right (it is the same "Traditional ML: texture features + SVM"
family already identified in the Literature Review slide) -- this script
trains an HOG-feature + SVM classifier on:
    (A) raw grayscale images
    (B) CLAHE + Gaussian-blurred images (this project's preprocessing pipeline)
and reports Accuracy / Precision / Recall / F1 for both, using the
dataset's existing train/validation split.
"""

import os
import json
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

import sys
sys.path.append(os.path.dirname(__file__))
from preprocessing import load_grayscale, preprocess_for_classifier, apply_canny

import cv2

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

TRAIN_DIR = "../data/train/images"
VAL_DIR = "../data/validation/images"
TARGET_SIZE = (128, 128)


def list_image_paths(base_dir):
    items = []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(base_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(".jpg"):
                items.append((os.path.join(cls_dir, fname), label_idx))
    return items


def extract_hog_raw(path):
    img = load_grayscale(path)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    feat = hog(img, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys")
    return feat


def extract_hog_preprocessed(path):
    """
    Matches the pitch deck's full pipeline: CLAHE -> Gaussian Blur -> Canny.
    HOG is computed on the CLAHE+blurred image (texture/intensity features)
    and concatenated with HOG computed on the Canny edge map (edge-geometry
    features), so all three pitched preprocessing stages feed the classifier
    -- not just the first two.
    """
    blurred = preprocess_for_classifier(path, target_size=TARGET_SIZE)
    edges = apply_canny(blurred)

    feat_blurred = hog(blurred, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), block_norm="L2-Hys")
    feat_edges = hog(edges, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), block_norm="L2-Hys")
    return np.concatenate([feat_blurred, feat_edges])


def build_feature_matrix(items, extractor):
    X, y = [], []
    for path, label in items:
        X.append(extractor(path))
        y.append(label)
    return np.array(X), np.array(y)


def run_condition(name, extractor, train_items, val_items):
    print(f"\n=== Condition: {name} ===")
    X_train, y_train = build_feature_matrix(train_items, extractor)
    X_val, y_val = build_feature_matrix(val_items, extractor)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_val_s)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=CLASSES, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(report)

    return {
        "condition": name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }


def main():
    train_items = list_image_paths(TRAIN_DIR)
    val_items = list_image_paths(VAL_DIR)
    print(f"Train images: {len(train_items)} | Validation images: {len(val_items)}")

    results = []
    results.append(run_condition("Raw -> HOG -> SVM", extract_hog_raw, train_items, val_items))
    results.append(run_condition("Preprocessed (CLAHE+Blur+Canny) -> HOG -> SVM", extract_hog_preprocessed, train_items, val_items))

    os.makedirs("../outputs", exist_ok=True)
    with open("../outputs/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: ../outputs/baseline_results.json")


if __name__ == "__main__":
    main()
