"""
Visual demonstration for the final ResNet50 steel-defect classifier.

This script:
1. Loads the final trained checkpoint.
2. Selects one validation image from each class.
3. Runs inference on the selected images.
4. Displays the true class, predicted class, and confidence.
5. Saves the visualization to outputs/demo/demo_predictions_raw.png.

No training is performed.

Image loading matches NEUDataset (resnet50_transfer.py) exactly:
grayscale read via OpenCV -> resized to 224x224 -> replicated into 3
channels. This keeps the demo's inference conditions identical to the
conditions used for every reported result in this project (training,
hyperparameter tuning, and evaluate_model.py all resize to 224x224
before creating tensors; using a different loader/size here would mean
the demo doesn't run under the same conditions as everything else).
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "final_results",
    "best_tuned_resnet50_raw.pth",
)

VALIDATION_IMAGES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "validation",
    "images",
)

DEMO_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "demo",
)

DEMO_OUTPUT_PATH = os.path.join(
    DEMO_OUTPUT_DIR,
    "demo_predictions_raw.png",
)

TARGET_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(num_classes):
    """Rebuild the ResNet50 architecture used during training."""

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def load_checkpoint(checkpoint_path, device):
    """Load the saved checkpoint."""

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    required_keys = [
        "model_state_dict",
        "class_names",
        "input_condition",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required keys: {missing_keys}"
        )

    return checkpoint


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def find_sample_image(class_directory):
    """Return the first valid image found in a class folder."""

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )

    image_names = sorted(
        filename
        for filename in os.listdir(class_directory)
        if filename.lower().endswith(valid_extensions)
    )

    if not image_names:
        raise FileNotFoundError(
            f"No image files found in: {class_directory}"
        )

    return os.path.join(
        class_directory,
        image_names[0],
    )


# ---------------------------------------------------------------------------
# Image loading (matches NEUDataset's "raw" branch exactly)
# ---------------------------------------------------------------------------

def load_and_prepare_image(image_path):
    """
    Loads an image the same way NEUDataset does for the "raw" condition:
    grayscale read -> resized to 224x224 -> replicated into 3 channels.

    Returns the 3-channel RGB array (for the model input) and a
    grayscale array (for display in the demo figure).
    """

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray_resized = cv2.resize(
        gray,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    rgb = cv2.cvtColor(
        gray_resized,
        cv2.COLOR_GRAY2RGB,
    )

    return rgb, gray_resized


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_image(
    model,
    image_path,
    transform,
    class_names,
    device,
):
    """Run inference on one image."""

    rgb_input, display_image = load_and_prepare_image(image_path)

    input_tensor = transform(rgb_input).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

    predicted_class = class_names[
        predicted_index.item()
    ]

    confidence_percentage = (
        confidence.item() * 100
    )

    return (
        display_image,
        predicted_class,
        confidence_percentage,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_demo_figure(results):
    """Create and save the prediction visualization."""

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    axes = axes.flatten()

    for axis, result in zip(
        axes,
        results,
    ):
        image = result["image"]
        true_class = result["true_class"]
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]

        is_correct = (
            true_class == predicted_class
        )

        result_text = (
            "Correct"
            if is_correct
            else "Incorrect"
        )

        axis.imshow(image, cmap="gray")
        axis.axis("off")

        axis.set_title(
            (
                f"Actual: {true_class}\n"
                f"Predicted: {predicted_class}\n"
                f"Confidence: {confidence:.2f}% | "
                f"{result_text}"
            ),
            fontsize=11,
        )

    figure.suptitle(
        "Steel Surface Defect Classification Demo",
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    figure.savefig(
        DEMO_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(figure)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    os.makedirs(
        DEMO_OUTPUT_DIR,
        exist_ok=True,
    )

    if not os.path.isfile(
        CHECKPOINT_PATH
    ):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not os.path.isdir(
        VALIDATION_IMAGES_DIR
    ):
        raise FileNotFoundError(
            "Validation images directory not found:\n"
            f"{VALIDATION_IMAGES_DIR}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        device,
    )

    class_names = list(
        checkpoint["class_names"]
    )

    print(
        "Checkpoint condition: "
        f"{checkpoint['input_condition']}"
    )

    print(
        "Classes: "
        f"{class_names}"
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

    model = build_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    results = []

    print("\nRunning demo predictions...\n")

    for true_class in class_names:
        class_directory = os.path.join(
            VALIDATION_IMAGES_DIR,
            true_class,
        )

        if not os.path.isdir(
            class_directory
        ):
            raise FileNotFoundError(
                "Class directory not found:\n"
                f"{class_directory}"
            )

        image_path = find_sample_image(
            class_directory
        )

        (
            image,
            predicted_class,
            confidence,
        ) = predict_image(
            model=model,
            image_path=image_path,
            transform=transform,
            class_names=class_names,
            device=device,
        )

        is_correct = (
            true_class == predicted_class
        )

        print(
            f"Actual: {true_class:18s} | "
            f"Predicted: {predicted_class:18s} | "
            f"Confidence: {confidence:6.2f}% | "
            f"{'Correct' if is_correct else 'Incorrect'}"
        )

        results.append(
            {
                "image": image,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
            }
        )

    create_demo_figure(
        results
    )

    correct_count = sum(
        result["true_class"]
        == result["predicted_class"]
        for result in results
    )

    print("\n" + "=" * 70)
    print("VISUAL DEMONSTRATION COMPLETE")
    print("=" * 70)

    print(
        f"Correct demo predictions: "
        f"{correct_count}/{len(results)}"
    )

    print(
        "\nDemo visualization saved to:\n"
        f"{DEMO_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()