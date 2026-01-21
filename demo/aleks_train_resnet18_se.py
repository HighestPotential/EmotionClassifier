import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from aleks_resnet18_se import ResNet18SE


CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_num_workers():
    c = os.cpu_count() or 8
    return max(2, min(8, c - 1))


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


def make_class_weights(counts):
    counts = torch.tensor(counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    w = inv / inv.sum() * len(counts)
    return w


def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def accuracy(model, loader, device, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total:
            pbar.set_postfix(acc=f"{(correct/total)*100:.2f}%")
    return correct / max(1, total)


def main():
    
    seed_everything(int(os.getenv("SEED", "42")))

    default_ready_root = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\ready_to_use_datasets"
    ready_root = os.getenv("READY_ROOT", default_ready_root)

    img_size = int(os.getenv("IMAGE_SIZE", "64"))
    num_classes = 6
    bs = int(os.getenv("BATCH_SIZE", "128"))
    epochs = int(os.getenv("EPOCHS", "300"))

    lr = float(os.getenv("LR", "0.03"))
    momentum = float(os.getenv("MOMENTUM", "0.9"))
    wd = float(os.getenv("WEIGHT_DECAY", "0.0005"))
    label_smooth = float(os.getenv("LABEL_SMOOTHING", "0.05"))
    mixup_alpha = float(os.getenv("MIXUP_ALPHA", "0.15"))
    mixup_prob = float(os.getenv("MIXUP_PROB", "0.3"))

    device = get_device()
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    num_workers = int(os.getenv("NUM_WORKERS", str(auto_num_workers() if device.type == "cuda" else 0)))
    pin_mem = (device.type == "cuda")
    persistent = (num_workers > 0)

    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)
        ], p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds_names = get_dataset_names(ready_root)

    train_ds = load_split_dataset(ready_root, ds_names, "train", train_tfm)
    eval_ds = load_split_dataset(ready_root, ds_names, "eval", eval_tfm)
    test_ds = load_split_dataset(ready_root, ds_names, "test", eval_tfm)
    
    print("Device:", device)
    print("READY_ROOT:", ready_root)
    print("Datasets:", ", ".join(ds_names))
    print("num_workers:", num_workers)
    print("train/eval/test:", len(train_ds), len(eval_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )

    model = ResNet18SE(num_classes=num_classes).to(device)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, num_classes)
        ).to(device)

    samples_per_cls = count_samples_per_class(train_ds)
    cls_weights = make_class_weights(samples_per_cls).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smooth).to(device)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_eval = 0.0
    best_train = 0.0
    best_path = "best_resnet18_se.pth"
    patience = 999
    best_ep = 0

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0

        pbar = tqdm(train_loader, desc=f"Train {ep}/{epochs}", unit="batch", leave=True)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            do_mixup = np.random.rand() < mixup_prob
            if do_mixup:
                imgs, y_a, y_b, lam = mixup_batch(imgs, labels, alpha=mixup_alpha)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    if do_mixup:
                        loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                    else:
                        loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(imgs)
                if do_mixup:
                    loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            batch_sz = labels.size(0)
            running_loss += loss.item() * batch_sz
            seen += batch_sz
            correct += (logits.argmax(1) == labels).sum().item()

            avg_loss = running_loss / max(1, seen)
            avg_acc = (correct / max(1, seen)) * 100.0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")

        sched.step()

        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        eval_acc = accuracy(model, eval_loader, device, desc=f"Eval  {ep}/{epochs}")

        print(
            f"Epoch {ep}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"eval_acc={eval_acc*100:.2f}%"
        )

        if eval_acc > best_eval:
            best_eval = eval_acc
            best_train = train_acc
            best_ep = ep
            torch.save(model.state_dict(), best_path)
        elif ep - best_ep >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)

    test_acc = accuracy(model, test_loader, device, desc="Test")

    print(f"Corresponding train acc: {best_train*100:.2f}%")
    print(f"Best eval acc: {best_eval*100:.2f}%")
    print(f"Test acc     : {test_acc*100:.2f}%")
    print(f"Saved to     : {best_path}")


if __name__ == "__main__":
    main()