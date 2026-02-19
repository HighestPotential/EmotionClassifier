# ----------------------------
# Standard libraries & imports
# ----------------------------
import os
import math
import random
import platform
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import Counter

# Custom dual-view ResNet18 variant
from F_aleks_resnet18_se_LReLUDual import ResNet18SE

# ----------------------------
# Classes and mapping
# ----------------------------
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

# ----------------------------
# Dual-view dataset for two inputs
# ----------------------------
class DualViewDataset(Dataset):
    """
    Combines two datasets containing different "views" of the same image.
    Returns: x1, x2, label
    """
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        x1, y1 = self.dataset1[idx]
        x2, y2 = self.dataset2[idx]
        assert y1 == y2
        return x1, x2, y1

# ----------------------------
# Compute class weights for imbalance handling
# ----------------------------
def compute_class_weights(dataset, num_classes):
    counter = Counter()
    for _, label in dataset:
        counter[label] += 1

    total = sum(counter.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / (counter[c] + 1e-6)
    return weights / weights.sum() * num_classes

# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Device selection (CPU/GPU/MPS)
# ----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ----------------------------
# Auto-set number of workers for DataLoader
# ----------------------------
def auto_num_workers():
    c = os.cpu_count() or 8
    return max(2, min(8, c - 1))

# ----------------------------
# Detect available dataset folders
# ----------------------------
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

# ----------------------------
# Dataset wrapper: remap classes to indices
# ----------------------------
class RemappedImageFolder(Dataset):
    """
    Similar to ImageFolder, but ensures consistent class-to-index mapping.
    """
    def __init__(self, root: str, transform, class_to_idx: dict[str, int]):
        self.root = root
        self.transform = transform
        self.class_to_idx = class_to_idx
        samples = []

        for class_name, target in class_to_idx.items():
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for dp, _, fns in os.walk(class_dir):
                for fn in fns:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}:
                        samples.append((os.path.join(dp, fn), target))
        if not samples:
            raise RuntimeError(f"No valid images found in: {root}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = datasets.folder.default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

# ----------------------------
# Load datasets for a specific split (train/eval/test)
# ----------------------------
def load_split_dataset(ready_root: str, dataset_names: list[str], split: str, transform):
    parts = []
    for name in dataset_names:
        split_path = os.path.join(ready_root, name, split)
        if os.path.isdir(split_path):
            parts.append(RemappedImageFolder(split_path, transform=transform, class_to_idx=CLASS_TO_IDX))
    if not parts:
        raise RuntimeError(f"No images found for split='{split}' in {ready_root}")
    return ConcatDataset(parts)

# ----------------------------
# Count samples per class
# ----------------------------
def count_samples_per_class(dataset):
    counts = [0] * len(CLASSES)
    def add_from_one(ds):
        if hasattr(ds, "samples"):
            for _, y in ds.samples:
                if 0 <= y < len(counts):
                    counts[y] += 1
        elif hasattr(ds, "targets"):
            for y in ds.targets:
                if 0 <= y < len(counts):
                    counts[y] += 1
        else:
            raise RuntimeError(f"Don't know how to count labels for dataset type: {type(ds)}")
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            add_from_one(ds)
    else:
        add_from_one(dataset)
    return counts

# ----------------------------
# Compute normalized inverse frequency weights
# ----------------------------
def make_class_weights(counts):
    counts = torch.tensor(counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    w = inv / inv.sum() * len(counts)
    return w

# ----------------------------
# Mixup augmentation helpers
# ----------------------------
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------
# Evaluation helpers
# ----------------------------
@torch.no_grad()
def accuracy(model, loader, device, desc="Eval"):
    """
    Computes top-1 accuracy on a given loader
    """
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x1, x2, labels in pbar:  # dual-view inputs
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        logits = model(x1, x2)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(acc=f"{100*correct/max(1,total):.2f}%")
    return correct / max(1, total)

@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    """
    Computes average loss over a loader
    """
    model.eval()
    total_loss = 0.0
    total = 0
    for x1,x2, labels in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(x1, x2)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(1, total)

# ----------------------------
# Class-Balanced Focal Loss
# ----------------------------
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        samples = torch.tensor(samples_per_class, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, samples)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples)
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ----------------------------
# Main training function
# ----------------------------
def main():
    # ----------------------------
    # Setup
    # ----------------------------
    seed_everything(int(os.getenv("SEED", "42")))
    default_ready_root = os.path.expanduser("/home/k/kienzlehagen/version_3/latest_3_0_ready_to_use_datasets")
    ready_root = os.getenv("READY_ROOT", default_ready_root)

    weights_dir = r"C:\...\weights"  # directory to save model weights
    os.makedirs(weights_dir, exist_ok=True)
    comparison_dir = r"C:\...\Comparison\SGD_class_balanced_focal_loss"
    os.makedirs(comparison_dir, exist_ok=True)
    results = []

    img_size = int(os.getenv("IMAGE_SIZE", "64"))
    num_classes = 6
    bs = int(os.getenv("BATCH_SIZE", "128"))
    epochs = int(os.getenv("EPOCHS", "200"))

    lr = float(os.getenv("LR", "0.03"))
    momentum = float(os.getenv("MOMENTUM", "0.9"))
    wd = float(os.getenv("WEIGHT_DECAY", "0.0005"))
    mixup_alpha = float(os.getenv("MIXUP_ALPHA", "0.15"))
    mixup_prob = float(os.getenv("MIXUP_PROB", "0.3"))

    device = get_device()
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    # workers for DataLoader
    num_workers = int(os.getenv("NUM_WORKERS", str(auto_num_workers() if device.type == "cuda" else 0)))
    pin_mem = (device.type == "cuda")
    persistent = (num_workers > 0)
    if platform.system() == "Darwin":  # disable workers on MacOS
        num_workers = 0
        persistent = False

    # ----------------------------
    # Transforms
    # ----------------------------
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.00)
        ], p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        transforms.RandomErasing(p=0.2, scale=(0.02,0.12), ratio=(0.3,3.3), value=0.0),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    # ----------------------------
    # Load dual-view datasets
    # ----------------------------
    ds_names = get_dataset_names(ready_root)
    train_ds_view1 = load_split_dataset(ready_root, ds_names, "train", eval_tfm)
    train_ds_view2 = load_split_dataset(ready_root, ds_names, "train", train_tfm)
    train_ds = DualViewDataset(train_ds_view1, train_ds_view2)  # dual-view

    eval_ds_1 = load_split_dataset(ready_root, ds_names, "eval", eval_tfm)
    eval_ds_2 = load_split_dataset(ready_root, ds_names, "eval", eval_tfm)
    eval_ds = DualViewDataset(eval_ds_1, eval_ds_2)

    test_ds_1 = load_split_dataset(ready_root, ds_names, "test", eval_tfm)
    test_ds_2 = load_split_dataset(ready_root, ds_names, "test", eval_tfm)
    test_ds = DualViewDataset(test_ds_1, test_ds_2)

    # ----------------------------
    # DataLoaders
    # ----------------------------
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, persistent_workers=persistent, prefetch_factor=2 if persistent else None)
    eval_loader = DataLoader(eval_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, persistent_workers=persistent, prefetch_factor=2 if persistent else None)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, persistent_workers=persistent, prefetch_factor=2 if persistent else None)

    # ----------------------------
    # Model
    # ----------------------------
    model = ResNet18SE(num_classes=num_classes).to(device)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, num_classes)
        ).to(device)

    # Class weights & loss
    samples_per_cls = count_samples_per_class(train_ds_view1)
    cls_weights = make_class_weights(samples_per_cls).to(device)
    criterion = ClassBalancedFocalLoss(samples_per_class=samples_per_cls, beta=0.9999, gamma=2.0).to(device)

    # Optimizer and scheduler
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    # AMP
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_eval = 0.0
    best_train = 0.0
    best_path = os.path.join(weights_dir, "best_FAleksLRELUDUAL.pth")
    patience = 999
    best_ep = 0

    # ----------------------------
    # Training loop
    # ----------------------------
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"Train {ep}/{epochs}", unit="batch", leave=True)
        for x1, x2, labels in pbar:  # dual-view batch
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            do_mixup = np.random.rand() < mixup_prob
            if do_mixup:  # mixup for both views
                x1, y_a, y_b, lam = mixup_batch(x1, labels, alpha=mixup_alpha)
                x2, _, _, _ = mixup_batch(x2, labels, alpha=mixup_alpha)

            # Forward + loss
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x1, x2)
                    loss = mixup_loss(criterion, logits, y_a, y_b, lam) if do_mixup else criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(x1, x2)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam) if do_mixup else criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            batch_sz = labels.size(0)
            running_loss += loss.item() * batch_sz
            seen += batch_sz
            correct += (logits.argmax(1) == labels).sum().item()
            pbar.set_postfix(loss=f"{running_loss/max(1,seen):.4f}", acc=f"{100*correct/max(1,seen):.2f}%")

        sched.step()
        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        eval_acc = accuracy(model, eval_loader, device, desc=f"Eval {ep}/{epochs}")
        eval_loss_val = eval_loss(model, eval_loader, criterion, device)

        if ep % 5 == 0:
            results.append({"epoch": ep, "train_loss": train_loss, "eval_loss": eval_loss_val, "train_acc": train_acc, "eval_acc": eval_acc})

        print(f"Epoch {ep}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | eval_acc={eval_acc*100:.2f}%")

        if eval_acc > best_eval:  # save best model
            best_eval = eval_acc
            best_train = train_acc
            best_ep = ep
            torch.save(model.state_dict(), best_path)
        elif ep - best_ep >= patience:
            print(f"Early stopping at epoch {ep}")
            break

  
