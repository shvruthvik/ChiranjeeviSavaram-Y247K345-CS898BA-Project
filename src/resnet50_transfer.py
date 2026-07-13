"""
ResNet50 transfer-learning classifier for NEU-DET steel surface defects.

STATUS: NOT YET EXECUTED. See README "Roadblocks & Pivots".
This project's development sandbox has no GPU, no PyTorch/TensorFlow
installed, and no internet access to fetch ImageNet-pretrained weights.
This script is complete and ready to run in any environment with those
three things available (e.g. Google Colab, a local GPU machine, or a
university compute cluster) -- run it there and drop the resulting
metrics into the README's results table for Milestone 3.

Usage:
    python resnet50_transfer.py --data_dir ../data --condition raw
    python resnet50_transfer.py --data_dir ../data --condition preprocessed

Requires: torch, torchvision, scikit-learn, opencv-python
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_grayscale, preprocess_multichannel_for_resnet

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


class NEUDataset(Dataset):
    def __init__(self, base_dir, condition, transform):
        self.items = []
        self.condition = condition
        self.transform = transform
        for label_idx, cls in enumerate(CLASSES):
            cls_dir = os.path.join(base_dir, "images", cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(".jpg"):
                    self.items.append((os.path.join(cls_dir, fname), label_idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        if self.condition == "raw":
            gray = load_grayscale(path)
            gray = cv2.resize(gray, (224, 224))
            # ResNet50 expects 3-channel input; replicate grayscale -> RGB
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            # Matches the pitch deck pipeline exactly: CLAHE -> Gaussian Blur
            # -> Canny, with each stage encoded as its own channel (see
            # preprocessing.preprocess_multichannel_for_resnet docstring).
            rgb = preprocess_multichannel_for_resnet(path, target_size=(224, 224))

        tensor = self.transform(rgb)
        return tensor, label


def build_model(num_classes=6):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze final block + classifier head for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(model, train_loader, val_loader, device, epochs=15, lr=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device, return_metrics=False)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    return model


def evaluate(model, loader, device, return_metrics=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    if not return_metrics:
        return acc

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--condition", choices=["raw", "preprocessed"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | condition={args.condition}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    train_ds = NEUDataset(os.path.join(args.data_dir, "train"), args.condition, transform)
    val_ds = NEUDataset(os.path.join(args.data_dir, "validation"), args.condition, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(num_classes=len(CLASSES))
    model = train(model, train_loader, val_loader, device, epochs=args.epochs)

    metrics = evaluate(model, val_loader, device)
    print(f"\nFinal validation metrics ({args.condition}):")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
