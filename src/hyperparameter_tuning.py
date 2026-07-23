"""
Hyperparameter optimization for the ResNet50 steel-defect classifier.

The script evaluates multiple learning rates, batch sizes, and
fine-tuning strategies using the NEU-DET dataset.

The default input condition is raw grayscale because the raw ResNet50
condition performed slightly better than the preprocessed condition
during Milestone 2. A --condition argument is still provided so that
the same experiment can be repeated using preprocessed inputs.

During the tuning sweep, validation accuracy is used to select the best
configuration. A separate final evaluation script should be used later
to calculate precision, recall, F1-score, per-class metrics, and the
confusion matrix for the winning model.

Each experiment records:
- training loss
- validation accuracy
- best validation accuracy
- execution time
- selected hyperparameters

The best model checkpoint, JSON results, and comparison plot are saved
inside the Part 3 output folders.
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from resnet50_transfer import CLASSES, NEUDataset


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


def set_seed(seed=42):
    """
    Set random seeds to improve experiment reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_tuning_model(
    num_classes,
    fine_tuning_strategy,
):
    """
    Create an ImageNet-pretrained ResNet50 model.

    Supported strategies:

    fc_only:
        Freeze all pretrained layers and train only the new
        six-class classification layer.

    layer4_fc:
        Fine-tune the final residual block and the new
        classification layer.
    """

    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    if fine_tuning_strategy == "layer4_fc":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    elif fine_tuning_strategy != "fc_only":
        raise ValueError(
            "Unsupported fine-tuning strategy: "
            f"{fine_tuning_strategy}"
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def calculate_accuracy(
    model,
    loader,
    device,
):
    """
    Calculate classification accuracy for one dataset.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    if total == 0:
        raise ValueError(
            "The validation dataset contains no images."
        )

    return correct / total


def run_experiment(
    experiment,
    train_dataset,
    validation_dataset,
    device,
    epochs,
):
    """
    Train and validate one hyperparameter configuration.

    A separate model is initialized for every experiment. The random
    seed is reset before each experiment so that the configurations
    are compared under consistent initialization and data-shuffling
    conditions.
    """

    print("\n" + "=" * 70)
    print(f"Experiment: {experiment['name']}")
    print(
        "Learning rate: "
        f"{experiment['learning_rate']}"
    )
    print(
        "Batch size: "
        f"{experiment['batch_size']}"
    )
    print(
        "Fine-tuning strategy: "
        f"{experiment['fine_tuning_strategy']}"
    )
    print("=" * 70)

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=experiment["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = build_tuning_model(
        num_classes=len(CLASSES),
        fine_tuning_strategy=(
            experiment["fine_tuning_strategy"]
        ),
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=experiment["learning_rate"],
    )

    history = {
        "training_loss": [],
        "validation_accuracy": [],
    }

    best_validation_accuracy = 0.0
    best_epoch = 0
    best_model_state = None

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

        training_loss = (
            running_loss / len(train_loader.dataset)
        )

        validation_accuracy = calculate_accuracy(
            model,
            validation_loader,
            device,
        )

        history["training_loss"].append(
            training_loss
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"training_loss={training_loss:.4f} | "
            f"validation_accuracy="
            f"{validation_accuracy:.4f}"
        )

    execution_time_seconds = (
        time.time() - start_time
    )

    if best_model_state is None:
        raise RuntimeError(
            "No model checkpoint was created for "
            f"{experiment['name']}."
        )

    return {
        "name": experiment["name"],
        "learning_rate": (
            experiment["learning_rate"]
        ),
        "batch_size": experiment["batch_size"],
        "fine_tuning_strategy": (
            experiment["fine_tuning_strategy"]
        ),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "execution_time_seconds": (
            execution_time_seconds
        ),
        "history": history,
        "model_state": best_model_state,
    }


def create_comparison_plot(
    results,
    condition,
):
    """
    Create a validation-accuracy comparison bar chart.

    A fixed 0-to-100 percent scale is used so that small differences
    between high-performing configurations are not visually
    exaggerated.
    """

    names = [
        result["name"]
        for result in results
    ]

    accuracies = [
        result["best_validation_accuracy"] * 100
        for result in results
    ]

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        names,
        accuracies,
    )

    plt.ylabel(
        "Best Validation Accuracy (%)"
    )

    plt.xlabel(
        "Hyperparameter Configuration"
    )

    plt.title(
        "ResNet50 Hyperparameter Optimization "
        f"({condition.capitalize()} Input)"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar, accuracy in zip(
        bars,
        accuracies,
    ):
        label_height = min(
            accuracy + 1,
            98,
        )

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            label_height,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = os.path.join(
        FINAL_PLOTS_DIR,
        (
            "resnet50_hyperparameter_"
            f"comparison_{condition}.png"
        ),
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nComparison plot saved to:\n"
        f"{output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a ResNet50 hyperparameter optimization "
            "sweep for NEU-DET defect classification."
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
            "train and validation folders."
        ),
    )

    parser.add_argument(
        "--condition",
        choices=[
            "raw",
            "preprocessed",
        ],
        default="raw",
        help=(
            "Input condition used for tuning. Raw is the "
            "default because it performed better during "
            "Milestone 2."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help=(
            "Number of training epochs used for each "
            "hyperparameter configuration."
        ),
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be at least 1."
        )

    os.makedirs(
        FINAL_RESULTS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        FINAL_PLOTS_DIR,
        exist_ok=True,
    )

    set_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(
        f"Input condition: {args.condition}"
    )
    print(
        f"Epochs per experiment: {args.epochs}"
    )

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

    train_dataset = NEUDataset(
        os.path.join(
            args.data_dir,
            "train",
        ),
        condition=args.condition,
        transform=transform,
    )

    validation_dataset = NEUDataset(
        os.path.join(
            args.data_dir,
            "validation",
        ),
        condition=args.condition,
        transform=transform,
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "No training images were found. Check the "
            "--data_dir path and dataset structure."
        )

    if len(validation_dataset) == 0:
        raise ValueError(
            "No validation images were found. Check the "
            "--data_dir path and dataset structure."
        )

    print(
        f"Training images: {len(train_dataset)}"
    )

    print(
        "Validation images: "
        f"{len(validation_dataset)}"
    )

    experiments = [
        {
            "name": "Midterm Configuration",
            "learning_rate": 0.0001,
            "batch_size": 32,
            "fine_tuning_strategy": "layer4_fc",
        },
        {
            "name": "Lower Learning Rate",
            "learning_rate": 0.00005,
            "batch_size": 32,
            "fine_tuning_strategy": "layer4_fc",
        },
        {
            "name": "Higher Learning Rate",
            "learning_rate": 0.0002,
            "batch_size": 32,
            "fine_tuning_strategy": "layer4_fc",
        },
        {
            "name": "Smaller Batch",
            "learning_rate": 0.0001,
            "batch_size": 16,
            "fine_tuning_strategy": "layer4_fc",
        },
        {
            "name": "Classifier Only",
            "learning_rate": 0.0001,
            "batch_size": 32,
            "fine_tuning_strategy": "fc_only",
        },
    ]

    all_results = []
    best_result = None

    for experiment in experiments:
        set_seed(42)

        result = run_experiment(
            experiment,
            train_dataset,
            validation_dataset,
            device,
            epochs=args.epochs,
        )

        if (
            best_result is None
            or result["best_validation_accuracy"]
            > best_result[
                "best_validation_accuracy"
            ]
        ):
            best_result = result

        result_without_model = {
            key: value
            for key, value in result.items()
            if key != "model_state"
        }

        all_results.append(
            result_without_model
        )

    best_model_path = os.path.join(
        FINAL_RESULTS_DIR,
        (
            "best_tuned_resnet50_"
            f"{args.condition}.pth"
        ),
    )

    torch.save(
        {
            "model_state_dict": (
                best_result["model_state"]
            ),
            "class_names": CLASSES,
            "input_condition": args.condition,
            "learning_rate": (
                best_result["learning_rate"]
            ),
            "batch_size": (
                best_result["batch_size"]
            ),
            "fine_tuning_strategy": (
                best_result[
                    "fine_tuning_strategy"
                ]
            ),
            "epochs": args.epochs,
            "best_epoch": (
                best_result["best_epoch"]
            ),
            "best_validation_accuracy": (
                best_result[
                    "best_validation_accuracy"
                ]
            ),
            "random_seed": 42,
        },
        best_model_path,
    )

    summary = {
        "device": str(device),
        "random_seed": 42,
        "input_condition": args.condition,
        "selection_metric": (
            "best_validation_accuracy"
        ),
        "number_of_experiments": len(
            experiments
        ),
        "epochs_per_experiment": (
            args.epochs
        ),
        "tuning_decision": (
            "Raw is the default tuning condition because "
            "it performed slightly better than the "
            "preprocessed ResNet50 condition during "
            "Milestone 2. The condition can be changed "
            "with the --condition argument."
        ),
        "training_loop_note": (
            "The tuning script uses a separate training "
            "loop so that every experiment can be reset "
            "with the same random seed and tracked "
            "independently."
        ),
        "results": all_results,
        "best_configuration": {
            "name": best_result["name"],
            "learning_rate": (
                best_result["learning_rate"]
            ),
            "batch_size": (
                best_result["batch_size"]
            ),
            "fine_tuning_strategy": (
                best_result[
                    "fine_tuning_strategy"
                ]
            ),
            "best_epoch": (
                best_result["best_epoch"]
            ),
            "best_validation_accuracy": (
                best_result[
                    "best_validation_accuracy"
                ]
            ),
        },
    }

    results_path = os.path.join(
        FINAL_RESULTS_DIR,
        (
            "hyperparameter_tuning_"
            f"results_{args.condition}.json"
        ),
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    create_comparison_plot(
        all_results,
        args.condition,
    )

    print("\n" + "=" * 70)
    print(
        "HYPERPARAMETER OPTIMIZATION COMPLETE"
    )
    print("=" * 70)

    print(
        "Input condition: "
        f"{args.condition}"
    )

    print(
        "Best experiment: "
        f"{best_result['name']}"
    )

    print(
        "Best validation accuracy: "
        f"{best_result['best_validation_accuracy']:.4f}"
    )

    print(
        "Best epoch: "
        f"{best_result['best_epoch']}"
    )

    print(
        "Learning rate: "
        f"{best_result['learning_rate']}"
    )

    print(
        "Batch size: "
        f"{best_result['batch_size']}"
    )

    print(
        "Fine-tuning strategy: "
        f"{best_result['fine_tuning_strategy']}"
    )

    print(
        "\nResults saved to:\n"
        f"{results_path}"
    )

    print(
        "\nBest model saved to:\n"
        f"{best_model_path}"
    )


if __name__ == "__main__":
    main()