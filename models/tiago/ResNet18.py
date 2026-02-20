import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np
from matplotlib import pyplot as plt
from torch.optim import AdamW

from tqdm import tqdm
from torchvision import models
# from torchmetrics.classification import Accuracy
# from torchmetrics.classification import MulticlassAccuracy

from pathlib import Path


def main():
    # Root directory containing all prepared datasets
    ROOT = Path.home() / "version_3/latest_3_0_ready_to_use_datasets"
    print("=== SCRIPT STARTED ===", flush=True)

    # Select GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = 6  # Target emotion classes

    # Load pretrained ResNet18 backbone
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

    # Adapt first convolution for 64x64 input resolution
    # Replace 7x7 kernel (stride 2) with 3x3 (stride 1) to preserve spatial detail
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    # He initialization for modified first layer
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    # Remove initial max pooling to avoid aggressive early downsampling
    model.maxpool = nn.Identity()

    # Adapt final fully connected layer to 6 emotion classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Standard ImageNet normalization (consistent with pretrained weights)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Concatenate multiple training datasets into a single training corpus
    train_dataset = [
        datasets.ImageFolder(ROOT / "AffectNet" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "CKplusIm" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "EmoSet-118k" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "FERPlus" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "MMAFEDB" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "NHFI" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "NONAMEFormated" / "train", transform=transform),
        datasets.ImageFolder(ROOT / "RAF-DB" / "train", transform=transform),
    ]

    train_dataset = ConcatDataset(train_dataset)

    # DataLoader with shuffling for stochastic training
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Evaluation datasets (kept separate from training)
    test_dataset = [
        datasets.ImageFolder(ROOT / "AffectNet" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "CKplusIm" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "EmoSet-118k" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "FERPlus" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "MMAFEDB" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "NHFI" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "NONAMEFormated" / "eval", transform=transform),
        datasets.ImageFolder(ROOT / "RAF-DB" / "eval", transform=transform),
    ]

    test_dataset = ConcatDataset(test_dataset)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    # Directory for saving best-performing model
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    # Print class-to-index mappings for verification
    for d in train_dataset.datasets:
        print(d.class_to_idx)

    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Cross-entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    best_eval_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        running_loss = 0
        train_total = 0

        # -------- Training Loop --------
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        running_loss /= train_total

        # -------- Evaluation Loop --------
        model.eval()
        eval_correct, total = 0, 0
        eval_loss = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * labels.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                eval_correct += (predicted == labels).sum().item()

        eval_loss /= total
        eval_acc = 100.0 * eval_correct / total

        # Save model if validation accuracy improves
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(
                model.state_dict(),
                save_dir / "best_model18A3C2.pt"
            )

        print(
            f"Epoch {epoch+1} / "
            f"train_loss: {running_loss:.4f} / "
            f"train_acc: {train_acc:.2f}% / "
            f"eval_loss: {eval_loss:.2f} / "
            f"eval_acc: {eval_acc:.2f}"
        )

    # Report total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("ResNet18A3C2")


if __name__ == "__main__":
    main()
