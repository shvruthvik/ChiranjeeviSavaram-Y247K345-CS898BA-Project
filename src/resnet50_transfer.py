"""
ResNet50 transfer-learning classifier for NEU-DET steel surface defects.

This script compares two input conditions:

    1. Raw grayscale images replicated into three channels

    2. Preprocessed multichannel images containing:
       - CLAHE-enhanced grayscale
       - Gaussian-blurred grayscale
       - Canny edge map

The ImageNet-pretrained ResNet50 model is adapted for six NEU-DET
surface-defect classes. Most pretrained layers are frozen, while the
final residual block and classification layer are fine-tuned.

"""

import argparse
import os
import sys

import cv2
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# Allow importing preprocessing.py from the same src folder.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from preprocessing import (
    load_grayscale,
    preprocess_multichannel_for_resnet,
)


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


class NEUDataset(Dataset):
    def __init__(self, base_dir, condition, transform):
        self.items = []
        self.condition = condition
        self.transform = transform

        for label_idx, cls in enumerate(CLASSES):
            cls_dir = os.path.join(
                base_dir,
                "images",
                cls,
            )

            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(".jpg"):
                    self.items.append(
                        (
                            os.path.join(cls_dir, fname),
                            label_idx,
                        )
                    )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        if self.condition == "raw":
            gray = load_grayscale(path)

            gray = cv2.resize(
                gray,
                (224, 224),
                interpolation=cv2.INTER_AREA,
            )

            # ResNet50 requires three input channels, so the grayscale
            # image is replicated across three channels.
            model_input = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2RGB,
            )

        else:
            # Custom three-channel preprocessed representation:
            # channel 0: CLAHE-enhanced grayscale
            # channel 1: Gaussian-blurred CLAHE image
            # channel 2: Canny edge map
            model_input = preprocess_multichannel_for_resnet(
                path,
                target_size=(224, 224),
            )

        tensor = self.transform(model_input)

        return tensor, label


def build_model(num_classes=6):
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    # Freeze all pretrained parameters first.
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune the final ResNet block.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the ImageNet classifier with a six-class classifier.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=15,
    lr=1e-4,
):
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=lr,
    )

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

        train_loss = (
            running_loss / len(train_loader.dataset)
        )

        val_acc = evaluate(
            model,
            val_loader,
            device,
            return_metrics=False,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    return model


def evaluate(
    model,
    loader,
    device,
    return_metrics=True,
):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
                .cpu()
                .numpy()
            )

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    if not return_metrics:
        return accuracy

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            PROJECT_ROOT,
            "data",
        ),
    )

    parser.add_argument(
        "--condition",
        choices=[
            "raw",
            "preprocessed",
        ],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device} | "
        f"condition={args.condition}"
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

    train_ds = NEUDataset(
        os.path.join(
            args.data_dir,
            "train",
        ),
        args.condition,
        transform,
    )

    val_ds = NEUDataset(
        os.path.join(
            args.data_dir,
            "validation",
        ),
        args.condition,
        transform,
    )

    print(
        f"Training images: {len(train_ds)} | "
        f"Validation images: {len(val_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(
        num_classes=len(CLASSES)
    )

    model = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
    )

    metrics = evaluate(
        model,
        val_loader,
        device,
    )

    print(
        f"\nFinal validation metrics "
        f"({args.condition}):"
    )

    for metric_name, metric_value in metrics.items():
        if metric_name != "confusion_matrix":
            print(
                f"  {metric_name}: "
                f"{metric_value:.4f}"
            )


if __name__ == "__main__":
    main()