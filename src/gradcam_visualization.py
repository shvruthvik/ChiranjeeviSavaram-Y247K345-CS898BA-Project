"""
Generate Grad-CAM visualizations for the final ResNet50 defect classifier.

The script:
1. Loads the best tuned ResNet50 checkpoint.
2. Selects validation images from each defect class.
3. Runs model inference.
4. Generates Grad-CAM heatmaps from ResNet50 layer4.
5. Overlays the heatmaps on the original images.
6. Saves individual visualizations and a combined figure.

No model training is performed.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Project paths and constants
# ---------------------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "final_results",
    "best_tuned_resnet50_raw.pth",
)

DEFAULT_VALIDATION_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "validation",
    "images",
)

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "gradcam",
)

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


# ---------------------------------------------------------------------------
# Model construction and checkpoint loading
# ---------------------------------------------------------------------------

def build_evaluation_model(num_classes):
    """
    Rebuild the ResNet50 architecture used during training.

    ImageNet weights are not downloaded because all trained parameters
    are restored from the saved checkpoint.
    """

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def load_checkpoint(checkpoint_path, device):
    """
    Load the model checkpoint with compatibility across PyTorch versions.
    """

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
            "Checkpoint is missing required keys: "
            f"{missing_keys}"
        )

    return checkpoint


# ---------------------------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Generate Grad-CAM maps using activations and gradients from a target layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = (
            self.target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    def _save_activations(
        self,
        module,
        inputs,
        output,
    ):
        """
        Store the target layer's forward activations.
        """

        self.activations = output.detach()

    def _save_gradients(
        self,
        module,
        gradient_input,
        gradient_output,
    ):
        """
        Store gradients flowing out of the target layer.
        """

        self.gradients = gradient_output[0].detach()

    def generate(
        self,
        input_tensor,
        target_class=None,
    ):
        """
        Generate a normalized Grad-CAM heatmap for one image.
        """

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        predicted_class = int(
            output.argmax(dim=1).item()
        )

        if target_class is None:
            target_class = predicted_class

        if target_class < 0 or target_class >= output.shape[1]:
            raise ValueError(
                f"Target class {target_class} is outside the "
                f"valid range 0 to {output.shape[1] - 1}."
            )

        class_score = output[
            0,
            target_class,
        ]

        class_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Target-layer activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Target-layer gradients were not captured."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        weighted_activations = (
            weights * self.activations
        )

        heatmap = weighted_activations.sum(
            dim=1,
            keepdim=True,
        )

        heatmap = torch.relu(
            heatmap
        )

        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        heatmap = heatmap.squeeze().cpu().numpy()

        minimum = float(
            heatmap.min()
        )

        maximum = float(
            heatmap.max()
        )

        if maximum > minimum:
            heatmap = (
                heatmap - minimum
            ) / (
                maximum - minimum
            )
        else:
            heatmap = np.zeros_like(
                heatmap,
                dtype=np.float32,
            )

        return (
            heatmap,
            predicted_class,
            output.detach(),
        )

    def close(self):
        """
        Remove the registered PyTorch hooks.
        """

        self.forward_hook.remove()
        self.backward_hook.remove()


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def find_class_images(
    validation_dir,
    class_name,
):
    """
    Return all supported image paths inside one class directory.
    """

    class_directory = os.path.join(
        validation_dir,
        class_name,
    )

    if not os.path.isdir(
        class_directory
    ):
        raise FileNotFoundError(
            "Class directory not found:\n"
            f"{class_directory}"
        )

    image_paths = []

    for filename in sorted(
        os.listdir(class_directory)
    ):
        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.append(
                os.path.join(
                    class_directory,
                    filename,
                )
            )

    if not image_paths:
        raise ValueError(
            "No supported images were found in:\n"
            f"{class_directory}"
        )

    return image_paths


def prepare_image(
    image_path,
    transform,
    device,
):
    """
    Load an image and return its display image and normalized model tensor.
    """

    with Image.open(
        image_path
    ) as loaded_image:
        rgb_image = loaded_image.convert(
            "RGB"
        )

    display_image = np.asarray(
        rgb_image,
        dtype=np.float32,
    ) / 255.0

    input_tensor = transform(
        rgb_image
    ).unsqueeze(0).to(device)

    return (
        display_image,
        input_tensor,
    )


def select_correct_example(
    image_paths,
    expected_class,
    model,
    transform,
    device,
):
    """
    Find the first correctly classified image from a shuffled list.
    """

    shuffled_paths = image_paths.copy()

    random.shuffle(
        shuffled_paths
    )

    for image_path in shuffled_paths:
        display_image, input_tensor = prepare_image(
            image_path,
            transform,
            device,
        )

        with torch.no_grad():
            output = model(
                input_tensor
            )

            predicted_class = int(
                output.argmax(
                    dim=1
                ).item()
            )

        if predicted_class == expected_class:
            return (
                image_path,
                display_image,
                input_tensor,
            )

    raise RuntimeError(
        "No correctly classified example was found for "
        f"class index {expected_class}."
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def create_overlay(
    display_image,
    heatmap,
    alpha=0.45,
):
    """
    Overlay a Grad-CAM heatmap on the original image.
    """

    colormap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = colormap(
        heatmap
    )[:, :, :3]

    overlay = (
        (1.0 - alpha) * display_image
        + alpha * colored_heatmap
    )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


def save_individual_visualization(
    display_image,
    heatmap,
    overlay,
    true_class,
    predicted_class,
    confidence,
    filename,
    output_dir,
):
    """
    Save original image, heatmap, and overlay as one figure.
    """

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(
        display_image
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        "Heatmap Overlay"
    )

    for axis in axes:
        axis.axis(
            "off"
        )

    figure.suptitle(
        (
            f"True: {true_class} | "
            f"Predicted: {predicted_class} | "
            f"Confidence: {confidence:.2%}"
        ),
        fontsize=12,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    safe_class_name = true_class.replace(
        "-",
        "_",
    )

    output_path = os.path.join(
        output_dir,
        (
            f"gradcam_{safe_class_name}_"
            f"{os.path.splitext(filename)[0]}.png"
        ),
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def save_combined_figure(
    visualization_records,
    output_dir,
    condition,
):
    """
    Save one combined Grad-CAM figure containing every defect class.
    """

    number_of_rows = len(
        visualization_records
    )

    figure, axes = plt.subplots(
        number_of_rows,
        3,
        figsize=(
            12,
            3.5 * number_of_rows,
        ),
    )

    if number_of_rows == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    column_titles = [
        "Original Image",
        "Grad-CAM Heatmap",
        "Heatmap Overlay",
    ]

    for column_index, title in enumerate(
        column_titles
    ):
        axes[
            0,
            column_index,
        ].set_title(
            title,
            fontsize=12,
            fontweight="bold",
        )

    for row_index, record in enumerate(
        visualization_records
    ):
        axes[
            row_index,
            0,
        ].imshow(
            record["display_image"]
        )

        axes[
            row_index,
            1,
        ].imshow(
            record["heatmap"],
            cmap="jet",
            vmin=0,
            vmax=1,
        )

        axes[
            row_index,
            2,
        ].imshow(
            record["overlay"]
        )

        axes[
            row_index,
            0,
        ].set_ylabel(
            (
                f"True: {record['true_class']}\n"
                f"Predicted: {record['predicted_class']}\n"
                f"Confidence: {record['confidence']:.1%}"
            ),
            fontsize=10,
        )

        for column_index in range(
            3
        ):
            axes[
                row_index,
                column_index,
            ].axis(
                "off"
            )

    figure.suptitle(
        (
            "Grad-CAM Explanation of ResNet50 Predictions "
            f"({condition.capitalize()} Input)"
        ),
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    output_path = os.path.join(
        output_dir,
        (
            "gradcam_all_classes_"
            f"{condition}.png"
        ),
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM explanations for the "
            "final tuned ResNet50 model."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to the tuned ResNet50 checkpoint."
        ),
    )

    parser.add_argument(
        "--validation_dir",
        default=DEFAULT_VALIDATION_DIR,
        help=(
            "Path to the validation dataset containing "
            "one subfolder per class."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory in which Grad-CAM figures will be saved."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used when selecting examples."
        ),
    )

    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help=(
            "Heatmap opacity between 0 and 1."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError(
            "--overlay_alpha must be between 0 and 1."
        )

    if not os.path.isfile(
        args.checkpoint
    ):
        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{args.checkpoint}"
        )

    if not os.path.isdir(
        args.validation_dir
    ):
        raise FileNotFoundError(
            "Validation directory not found:\n"
            f"{args.validation_dir}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

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

    if condition != "raw":
        raise ValueError(
            "This Grad-CAM script currently expects a raw-input "
            "checkpoint so that overlays correspond directly to "
            "the original validation images. "
            f"Checkpoint condition was: {condition}"
        )

    print(
        f"Checkpoint condition: {condition}"
    )

    print(
        "Classes: "
        + ", ".join(
            class_names
        )
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

    model = build_evaluation_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    gradcam = GradCAM(
        model=model,
        target_layer=model.layer4[-1],
    )

    visualization_records = []

    try:
        for class_index, class_name in enumerate(
            class_names
        ):
            image_paths = find_class_images(
                args.validation_dir,
                class_name,
            )

            (
                selected_path,
                display_image,
                input_tensor,
            ) = select_correct_example(
                image_paths=image_paths,
                expected_class=class_index,
                model=model,
                transform=transform,
                device=device,
            )

            (
                heatmap,
                predicted_index,
                output,
            ) = gradcam.generate(
                input_tensor=input_tensor,
                target_class=None,
            )

            probabilities = torch.softmax(
                output,
                dim=1,
            )

            confidence = float(
                probabilities[
                    0,
                    predicted_index,
                ].item()
            )

            predicted_class = class_names[
                predicted_index
            ]

            overlay = create_overlay(
                display_image=display_image,
                heatmap=heatmap,
                alpha=args.overlay_alpha,
            )

            individual_path = (
                save_individual_visualization(
                    display_image=display_image,
                    heatmap=heatmap,
                    overlay=overlay,
                    true_class=class_name,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    filename=os.path.basename(
                        selected_path
                    ),
                    output_dir=args.output_dir,
                )
            )

            visualization_records.append(
                {
                    "display_image": display_image,
                    "heatmap": heatmap,
                    "overlay": overlay,
                    "true_class": class_name,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "image_path": selected_path,
                    "output_path": individual_path,
                }
            )

            print(
                (
                    f"{class_name}: "
                    f"{os.path.basename(selected_path)} -> "
                    f"{predicted_class} "
                    f"({confidence:.2%})"
                )
            )

            print(
                "  Saved: "
                f"{individual_path}"
            )

    finally:
        gradcam.close()

    combined_path = save_combined_figure(
        visualization_records=visualization_records,
        output_dir=args.output_dir,
        condition=condition,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GRAD-CAM GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Individual figures created: "
        f"{len(visualization_records)}"
    )

    print(
        "\nCombined Grad-CAM figure saved to:\n"
        f"{combined_path}"
    )


if __name__ == "__main__":
    main()