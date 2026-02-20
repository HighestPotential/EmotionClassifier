import os
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


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAM(planes, reduction=reduction, spatial_kernel=7)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNetCBAM(nn.Module):
    def __init__(self, block, layers, num_classes: int = 6, base_width: int = 64, reduction: int = 16):
        super().__init__()
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)

        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes: int, blocks: int, stride: int, reduction: int):
        layers = [block(self.in_planes, planes, stride=stride, reduction=reduction)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18CBAM(num_classes: int = 6, reduction: int = 16):
    return ResNetCBAM(BasicBlockCBAM, [2, 2, 2, 2], num_classes=num_classes, reduction=reduction)


def main():
    seed_everything(int(os.getenv("SEED", "42")))

    default_ready_root = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\ready_to_use_datasets"
    ready_root = os.getenv("READY_ROOT", default_ready_root)

    weights_dir = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\models\aleks\weights"
    os.makedirs(weights_dir, exist_ok=True)

    comparison_dir = r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\models\aleks\Comparison\SGD_class_balanced_focal_loss"
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

    num_workers = int(os.getenv("NUM_WORKERS", str(auto_num_workers() if device.type == "cuda" else 0)))
    pin_mem = (device.type == "cuda")
    persistent = (num_workers > 0)

    if platform.system() == "Darwin":
        num_workers = 0
        persistent = False

    train_tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)], p=0.5
            ),
            transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
        ]
    )

    eval_tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ds_names = get_dataset_names(ready_root)

    train_ds = load_split_dataset(ready_root, ds_names, "train", train_tfm)
    eval_ds = load_split_dataset(ready_root, ds_names, "eval", eval_tfm)
    test_ds = load_split_dataset(ready_root, ds_names, "test", eval_tfm)

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

    model = ResNet18CBAM(num_classes=num_classes).to(device)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, num_classes)).to(device)

    samples_per_cls = count_samples_per_class(train_ds)
    criterion = ClassBalancedFocalLoss(samples_per_class=samples_per_cls, beta=0.9999, gamma=2.0).to(device)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_eval = 0.0
    best_train = 0.0
    best_path = os.path.join(weights_dir, "best_resnet18_cbam.pth")
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
        eval_loss_val = eval_loss(model, eval_loader, criterion, device)

        if ep % 5 == 0:
            results.append(
                {
                    "epoch": ep,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss_val,
                    "train_acc": train_acc,
                    "eval_acc": eval_acc,
                }
            )

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

    csv_path = os.path.join(comparison_dir, "results_SGD_class_balanced_focal_loss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "eval_loss", "train_acc", "eval_acc"])
        writer.writeheader()
        writer.writerows(results)

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
    print(f"CSV saved to : {csv_path}")


if __name__ == "__main__":
    main()
