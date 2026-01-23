import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from aleks_resnet18_se import ResNet18SE


CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataset_names(ready_root: str):
    names = []
    if not os.path.isdir(ready_root):
        raise RuntimeError(f"ready_root does not exist: {ready_root}")
    for name in sorted(os.listdir(ready_root)):
        folder = os.path.join(ready_root, name)
        if not os.path.isdir(folder):
            continue
        ok = all(os.path.isdir(os.path.join(folder, split)) for split in ["train", "eval", "test"])
        if ok:
            names.append(name)
    if not names:
        raise RuntimeError(f"No datasets found in: {ready_root}")
    return names


class RemappedImageFolder(Dataset):
    def __init__(self, root: str, transform, class_to_idx: dict[str, int]):
        self.ds = datasets.ImageFolder(root=root, transform=transform)
        self.class_to_idx = class_to_idx

        mapped = []
        for path, local_y in self.ds.samples:
            class_name = self.ds.classes[local_y]
            if class_name not in self.class_to_idx:
                continue
            mapped.append((path, self.class_to_idx[class_name]))

        if not mapped:
            raise RuntimeError(f"No valid class folders found in: {root}")

        self.ds.samples = mapped
        self.ds.targets = [y for _, y in mapped]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def load_split_dataset(ready_root: str, dataset_names: list[str], split: str, transform):
    parts = []
    for name in dataset_names:
        split_path = os.path.join(ready_root, name, split)
        if os.path.isdir(split_path):
            parts.append(RemappedImageFolder(split_path, transform=transform, class_to_idx=CLASS_TO_IDX))
    if not parts:
        raise RuntimeError(f"No images found for split='{split}' in {ready_root}")
    return ConcatDataset(parts)


def count_samples_per_class(dataset):
    counts = [0] * len(CLASSES)
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            for y in ds.ds.targets:
                if 0 <= y < len(counts):
                    counts[y] += 1
    else:
        for y in dataset.ds.targets:
            if 0 <= y < len(counts):
                counts[y] += 1
    return counts


class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        counts = torch.tensor(samples_per_class, dtype=torch.float32)
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer("weights", weights)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


@torch.no_grad()
def accuracy(model, loader, device, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total:
            pbar.set_postfix(acc=f"{(correct/total)*100:.2f}%")
    return correct / max(1, total)


def main():
    default_ready_root = os.path.expanduser(
        "~/Desktop/Aleks Uni/Computer Vision/Final_Project/EmotionClassifier-main/ready_to_use_datasets"
    )
    ready_root = os.getenv("READY_ROOT", default_ready_root)

    image_size = 64
    num_classes = 6
    batch_size = 64
    epochs = 30
    lr = 0.05
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = int(os.getenv("NUM_WORKERS", "0"))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_names = get_dataset_names(ready_root)

    train_dataset = load_split_dataset(ready_root, dataset_names, "train", transform)
    eval_dataset = load_split_dataset(ready_root, dataset_names, "eval", transform)
    test_dataset = load_split_dataset(ready_root, dataset_names, "test", transform)

    device = get_device()
    print("Device:", device)

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = ResNet18SE(num_classes=num_classes).to(device)

    samples_per_class = count_samples_per_class(train_dataset)
    loss_fn = CBFocalLoss(samples_per_class=samples_per_class, beta=0.9999, gamma=2.0).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eval_acc = 0.0
    best_model_path = "best_resnet18_se.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        total_correct = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", unit="batch", leave=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_seen += bs
            total_correct += (logits.argmax(1) == labels).sum().item()

            running_loss = total_loss / max(1, total_seen)
            running_acc = (total_correct / max(1, total_seen)) * 100.0
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.2f}%")

        scheduler.step()

        train_loss = total_loss / max(1, total_seen)
        train_acc = total_correct / max(1, total_seen)
        eval_acc = accuracy(model, eval_loader, device, desc=f"Eval  {epoch}/{epochs}")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"eval_acc={eval_acc*100:.2f}%"
        )

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_acc = accuracy(model, test_loader, device, desc="Test")

    print(f"Best eval acc: {best_eval_acc*100:.2f}%")
    print(f"Test acc    : {test_acc*100:.2f}%")
    print(f"Saved to    : {best_model_path}")


if __name__ == "__main__":
    main()