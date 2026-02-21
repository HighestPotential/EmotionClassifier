import os
import math
import random
import platform
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# Constants
TRAIN_DATA_ROOT = "/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets"
TEST_DATA_ROOT = "/home/d/dumanskyy/work/EmotionClassifier/test_datasets"
WEIGHTS_DIR = "/home/d/dumanskyy/work/EmotionClassifier/models/aleks/improved_v1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Ensure imports work regardless of execution path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from aleks_resnet18_se import ResNet18SE
except ImportError:
    # Fallback if running directly from improved_v1
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # Assuming aleks_resnet18_se.py is in the parent folder, we already added it
    # If it is in the same folder, this works. If in parent:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
            # It's possible some subsets are empty, warn but don't crash unless all are empty
            pass 

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
            try:
                ds = RemappedImageFolder(split_path, transform=transform, class_to_idx=CLASS_TO_IDX)
                if len(ds) > 0:
                    parts.append(ds)
            except RuntimeError:
                pass
    if not parts:
        raise RuntimeError(f"No images found for split='{split}' in {ready_root}")
    return ConcatDataset(parts)

# --- Weighted Sampler Helpers ---
def count_samples_per_class(dataset):
    counts = [0] * len(CLASSES)

    def add_from_ds(ds):
        if isinstance(ds, ConcatDataset):
            for sub_ds in ds.datasets:
                add_from_ds(sub_ds)
        elif hasattr(ds, "samples"):
            for _, y in ds.samples:
                if 0 <= y < len(counts):
                    counts[y] += 1
        elif hasattr(ds, "targets"):
            for y in ds.targets:
                if 0 <= y < len(counts):
                    counts[y] += 1
        else:
            # Fallback for subsets or wrappers
            pass

    add_from_ds(dataset)
    return counts

def make_weighted_sampler(dataset):
    counts = count_samples_per_class(dataset)
    print(f"Class counts: {counts}")
    
    # Weight = 1 / count
    weights = []
    total = sum(counts)
    for c in counts:
        if c > 0:
            weights.append(total / c)
        else:
            weights.append(0.0)
            
    # Normalize weights so they sum to num_classes or similar (optional but good for debug)
    # Actually for the sampler we need weight per sample
    
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)
    
    sample_weights = []
    
    all_labels = []
    
    def collect_labels(ds):
        if isinstance(ds, ConcatDataset):
            for sub_ds in ds.datasets:
                collect_labels(sub_ds)
        elif hasattr(ds, "samples"):
            for _, y in ds.samples:
                all_labels.append(y)
        elif hasattr(ds, "targets"):
            all_labels.extend(ds.targets)
            
    collect_labels(dataset)
        
    # Map labels to weights
    sample_weights = [weights[y] for y in all_labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(sample_weights_tensor, len(sample_weights), replacement=True)
    return sampler, class_weights_tensor


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


@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(1, total)


def cosine_scheduler_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, min_lr=0.0):
    # Standard Cosine Annealing from PyTorch
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_training_epochs - num_warmup_epochs, 
        eta_min=min_lr
    )
    
    # Linear Warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=num_warmup_epochs
    )
    
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[num_warmup_epochs]
    )


def main():

    seed_everything(int(os.getenv("SEED", "42")))
    
    # Directories
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # CSV Result File
    csv_filename = "aleks_train_resnet18_se_SGD_cbfl_results.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    
    # Initialize CSV if not exists
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "LR", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])

    # Hyperparams
    img_size = int(os.getenv("IMAGE_SIZE", "64"))
    num_classes = 6
    bs = int(os.getenv("BATCH_SIZE", "128"))
    epochs = int(os.getenv("EPOCHS", "600"))

    lr = float(os.getenv("LR", "0.03"))
    momentum = float(os.getenv("MOMENTUM", "0.9"))
    wd = float(os.getenv("WEIGHT_DECAY", "0.0005"))
    mixup_alpha = float(os.getenv("MIXUP_ALPHA", "0.15"))
    mixup_prob = float(os.getenv("MIXUP_PROB", "0.3"))
    warmup_epochs = 5

    device = get_device()
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    num_workers = int(os.getenv("NUM_WORKERS", str(auto_num_workers() if device.type == "cuda" else 0)))
    pin_mem = (device.type == "cuda")
    persistent = (num_workers > 0)

    if platform.system() == "Darwin":
        num_workers = 0
        persistent = False

    # Augmentations (Removed RandomErasing)
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)
        ], p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # RandomErasing removed for faces
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds_names = get_dataset_names(TRAIN_DATA_ROOT)

    # Merge Train and Eval for Training
    train_ds_part1 = load_split_dataset(TRAIN_DATA_ROOT, ds_names, "train", train_tfm)
    train_ds_part2 = load_split_dataset(TRAIN_DATA_ROOT, ds_names, "eval", train_tfm)
    train_ds = ConcatDataset([train_ds_part1, train_ds_part2])
    
    # Use Test folder for Validation (Eval)
    eval_ds = load_split_dataset(TRAIN_DATA_ROOT, ds_names, "test", eval_tfm)

    # Weighted Random Sampler
    print("Calculating sampler weights...")
    sampler, _ = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=False, # Must be False for Sampler
        sampler=sampler,
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

    model = ResNet18SE(num_classes=num_classes).to(device)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, num_classes)
        ).to(device)

    # Replace CBFL with Standard CrossEntropy
    criterion = nn.CrossEntropyLoss().to(device)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    
    # Warmup Scheduler
    sched = cosine_scheduler_with_warmup(optim, warmup_epochs, epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_eval = 0.0
    best_train = 0.0
    best_path = os.path.join(WEIGHTS_DIR, "best_resnet18_se.pth")
    checkpoint_path = os.path.join(WEIGHTS_DIR, "checkpoint.pth")
    
    start_epoch = 1

    # Resume Logic
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Test run')
    args = parser.parse_args()

    if args.resume and os.path.isfile(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])
        sched.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_eval = ckpt.get('best_eval', 0.0)
        print(f"Resumed at epoch {start_epoch} with Best Eval Acc: {best_eval:.2f}%")

    if args.dry_run:
        epochs = 2
        print("Dry Run Enabled: 2 Epochs")

    for ep in range(start_epoch, epochs + 1):
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
        
        # Validation every epoch
        eval_acc = accuracy(model, eval_loader, device, desc=f"Eval  {ep}/{epochs}")
        eval_loss_val = eval_loss(model, eval_loader, criterion, device)
        
        current_lr = optim.param_groups[0]['lr']

        # Log to CSV every 5 epochs OR last epoch
        if ep % 5 == 0 or ep == epochs:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, current_lr, train_loss, train_acc, eval_loss_val, eval_acc])

        print(
            f"Epoch {ep}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"eval_acc={eval_acc*100:.2f}%"
        )

        # Save Checkpoint
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_eval': best_eval
        }, checkpoint_path)

        # Save Best
        if eval_acc > best_eval:
            best_eval = eval_acc
            best_train = train_acc
            best_ep = ep
            torch.save(model.state_dict(), best_path)
            print(f"New Best Model Saved! Acc: {best_eval*100:.2f}%")

    # --- Final Test Evaluation ---
    print("\nTraining Complete. Loading Best Model for Final Test...")
    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)

    # Load Test Datasets
    final_test_set_names = get_dataset_names(TEST_DATA_ROOT)
    final_test_ds = load_split_dataset(TEST_DATA_ROOT, final_test_set_names, "test", eval_tfm) # Assuming structure matches
    # If test datasets folder structure is just class folders directly inside dataset folders without 'test' split:
    # Logic might need adjustment. Assuming standard split structure.

    final_test_loader = DataLoader(
        final_test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    test_acc = accuracy(model, final_test_loader, device, desc="Final Test")
    print(f"Final Test Acc on {TEST_DATA_ROOT}: {test_acc*100:.2f}%")
    
    # Log Final Result
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FINAL_TEST", "-", "-", "-", "-", test_acc])


if __name__ == "__main__":
    main()
