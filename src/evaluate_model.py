"""
Final evaluation script for the tuned ResNet50 steel-defect classifier.

This script:
1. Loads the best checkpoint created by hyperparameter_tuning.py.
2. Rebuilds the ResNet50 architecture without downloading pretrained weights.
3. Evaluates the saved model on the NEU-DET validation dataset.
4. Calculates accuracy, macro and weighted precision, recall, and F1-score.
5. Generates a per-class classification report.
6. Saves a confusion-matrix plot.
7. Saves the complete evaluation results as JSON.

No additional training is performed.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from resnet50_transfer import NEUDataset


FINAL_RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "final_results",
)

FINAL_PLOTS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "final_plots",
)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_evaluation_model(num_classes):
    """
    Rebuild the ResNet50 architecture used during training.

    Pretrained ImageNet weights are not downloaded because all model
    parameters are loaded from the saved checkpoint.

    The fine-tuning strategy is not needed during evaluation because
    no model parameters are updated.
    """

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path, device):
    """
    Load a model checkpoint with compatibility for different PyTorch versions.
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    required_keys = [
        "model_state_dict",
        "class_names",
        "input_condition",
        "fine_tuning_strategy",
        "batch_size",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{missing_keys}"
        )

    return checkpoint


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def collect_predictions(
    model,
    data_loader,
    device,
):
    """
    Run inference and return the true and predicted class labels.
    """

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(
                labels.cpu().numpy().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().numpy().tolist()
            )

    if not true_labels:
        raise ValueError(
            "The evaluation dataset contains no images."
        )

    return true_labels, predicted_labels


# ---------------------------------------------------------------------------
# Confusion-matrix plot
# ---------------------------------------------------------------------------

def create_confusion_matrix_plot(
    matrix,
    class_names,
    condition,
):
    """
    Create and save the final confusion-matrix visualization.
    """

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    tick_positions = np.arange(
        len(class_names)
    )

    axis.set(
        xticks=tick_positions,
        yticks=tick_positions,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Class",
        ylabel="True Class",
        title=(
            "Final ResNet50 Confusion Matrix "
            f"({condition.capitalize()} Input)"
        ),
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size > 0
        else 0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    figure.tight_layout()

    output_path = os.path.join(
        FINAL_PLOTS_DIR,
        (
            "final_resnet50_confusion_matrix_"
            f"{condition}.png"
        ),
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best tuned ResNet50 model "
            "on the NEU-DET validation dataset."
        )
    )

    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            PROJECT_ROOT,
            "data",
        ),
        help=(
            "Path to the dataset directory containing "
            "the validation folder."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            FINAL_RESULTS_DIR,
            "best_tuned_resnet50_raw.pth",
        ),
        help=(
            "Path to the tuned ResNet50 checkpoint."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Optional evaluation batch size. When omitted, "
            "the batch size stored in the checkpoint is used."
        ),
    )

    args = parser.parse_args()

    os.makedirs(
        FINAL_RESULTS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        FINAL_PLOTS_DIR,
        exist_ok=True,
    )

    if not os.path.isfile(
        args.checkpoint
    ):
        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{args.checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    class_names = list(
        checkpoint["class_names"]
    )

    condition = checkpoint[
        "input_condition"
    ]

    fine_tuning_strategy = checkpoint[
        "fine_tuning_strategy"
    ]

    evaluation_batch_size = (
        args.batch_size
        if args.batch_size is not None
        else checkpoint["batch_size"]
    )

    if evaluation_batch_size < 1:
        raise ValueError(
            "Evaluation batch size must be at least 1."
        )

    print(
        "Checkpoint condition: "
        f"{condition}"
    )

    print(
        "Fine-tuning strategy: "
        f"{fine_tuning_strategy}"
    )

    print(
        "Training learning rate: "
        f"{checkpoint.get('learning_rate', 'Unknown')}"
    )

    print(
        "Training batch size: "
        f"{checkpoint.get('batch_size', 'Unknown')}"
    )

    print(
        "Best training epoch: "
        f"{checkpoint.get('best_epoch', 'Unknown')}"
    )

    print(
        "Stored best validation accuracy: "
        f"{checkpoint.get('best_validation_accuracy', 'Unknown')}"
    )

    print(
        "Evaluation batch size: "
        f"{evaluation_batch_size}"
    )

    # Use the same normalization applied during hyperparameter tuning.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    validation_path = os.path.join(
        args.data_dir,
        "validation",
    )

    validation_dataset = NEUDataset(
        validation_path,
        condition=condition,
        transform=transform,
    )

    if len(validation_dataset) == 0:
        raise ValueError(
            "No validation images were found. "
            "Check the --data_dir path and dataset structure."
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Validation images: "
        f"{len(validation_dataset)}"
    )

    model = build_evaluation_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)

    true_labels, predicted_labels = (
        collect_predictions(
            model,
            validation_loader,
            device,
        )
    )

    label_indices = list(
        range(len(class_names))
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average="weighted",
        zero_division=0,
    )

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_indices,
    )

    true_array = np.asarray(
        true_labels
    )

    predicted_array = np.asarray(
        predicted_labels
    )

    correct_predictions = int(
        np.sum(
            true_array == predicted_array
        )
    )

    incorrect_predictions = int(
        np.sum(
            true_array != predicted_array
        )
    )

    confusion_matrix_path = (
        create_confusion_matrix_plot(
            matrix,
            class_names,
            condition,
        )
    )

    results = {
        "checkpoint_path": os.path.abspath(
            args.checkpoint
        ),
        "validation_dataset_path": os.path.abspath(
            validation_path
        ),
        "device": str(device),
        "input_condition": condition,
        "fine_tuning_strategy": (
            fine_tuning_strategy
        ),
        "learning_rate": checkpoint.get(
            "learning_rate"
        ),
        "training_batch_size": checkpoint.get(
            "batch_size"
        ),
        "evaluation_batch_size": (
            evaluation_batch_size
        ),
        "training_epochs": checkpoint.get(
            "epochs"
        ),
        "best_epoch": checkpoint.get(
            "best_epoch"
        ),
        "random_seed": checkpoint.get(
            "random_seed"
        ),
        "stored_best_validation_accuracy": (
            checkpoint.get(
                "best_validation_accuracy"
            )
        ),
        "number_of_evaluation_images": len(
            true_labels
        ),
        "correct_predictions": (
            correct_predictions
        ),
        "incorrect_predictions": (
            incorrect_predictions
        ),
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1_score": float(
            macro_f1
        ),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1_score": float(
            weighted_f1
        ),
        "class_names": class_names,
        "classification_report": (
            report_dictionary
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
    }

    results_path = os.path.join(
        FINAL_RESULTS_DIR,
        (
            "final_evaluation_results_"
            f"{condition}.json"
        ),
    )

    report_path = os.path.join(
        FINAL_RESULTS_DIR,
        (
            "final_classification_report_"
            f"{condition}.txt"
        ),
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            results,
            output_file,
            indent=4,
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "Final ResNet50 Classification Report\n"
        )

        output_file.write(
            "=" * 50 + "\n\n"
        )

        output_file.write(
            f"Input condition: {condition}\n"
        )

        output_file.write(
            "Fine-tuning strategy: "
            f"{fine_tuning_strategy}\n"
        )

        output_file.write(
            "Learning rate: "
            f"{checkpoint.get('learning_rate', 'Unknown')}\n"
        )

        output_file.write(
            "Best epoch: "
            f"{checkpoint.get('best_epoch', 'Unknown')}\n"
        )

        output_file.write(
            "Evaluation images: "
            f"{len(true_labels)}\n"
        )

        output_file.write(
            "Correct predictions: "
            f"{correct_predictions}\n"
        )

        output_file.write(
            "Incorrect predictions: "
            f"{incorrect_predictions}\n\n"
        )

        output_file.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        output_file.write(
            "Macro precision: "
            f"{macro_precision:.4f}\n"
        )

        output_file.write(
            "Macro recall: "
            f"{macro_recall:.4f}\n"
        )

        output_file.write(
            "Macro F1-score: "
            f"{macro_f1:.4f}\n"
        )

        output_file.write(
            "Weighted precision: "
            f"{weighted_precision:.4f}\n"
        )

        output_file.write(
            "Weighted recall: "
            f"{weighted_recall:.4f}\n"
        )

        output_file.write(
            "Weighted F1-score: "
            f"{weighted_f1:.4f}\n\n"
        )

        output_file.write(
            "Per-Class Classification Report\n"
        )

        output_file.write(
            "-" * 50 + "\n"
        )

        output_file.write(
            report_text
        )

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        "Macro precision: "
        f"{macro_precision:.4f}"
    )

    print(
        "Macro recall: "
        f"{macro_recall:.4f}"
    )

    print(
        "Macro F1-score: "
        f"{macro_f1:.4f}"
    )

    print(
        "Weighted precision: "
        f"{weighted_precision:.4f}"
    )

    print(
        "Weighted recall: "
        f"{weighted_recall:.4f}"
    )

    print(
        "Weighted F1-score: "
        f"{weighted_f1:.4f}"
    )

    print(
        "Correct predictions: "
        f"{correct_predictions}"
    )

    print(
        "Incorrect predictions: "
        f"{incorrect_predictions}"
    )

    print("\nClassification Report:\n")
    print(report_text)

    print(
        "\nEvaluation results saved to:\n"
        f"{results_path}"
    )

    print(
        "\nClassification report saved to:\n"
        f"{report_path}"
    )

    print(
        "\nConfusion matrix saved to:\n"
        f"{confusion_matrix_path}"
    )


if __name__ == "__main__":
    main()