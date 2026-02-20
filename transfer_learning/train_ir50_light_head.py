import sys
import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import numpy as np
from torch.nn import BatchNorm2d, BatchNorm1d, Dropout, AdaptiveAvgPool2d
from tqdm import tqdm
from timm.utils import ModelEmaV2
from timm.scheduler import CosineLRScheduler

# Import IR50 backbone (reuses all block definitions from ir50.py)
try:
    from ir50 import Backbone, load_pretrained_weights, Flatten
except ImportError:
    print("Error: Could not import ir50.py. Ensure it is in the same directory.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
ACCUM_STEPS = 2
EPOCHS = 300
LR = 3e-4              # body4 backbone LR (gentle but meaningful fine-tuning)
HEAD_LR = 0.01         # head LR (~33× faster — random init needs to catch up)
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
IMG_SIZE = 112             # IR50 native resolution
NUM_WORKERS = 8
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Paths
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets'
TEST_DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/test_datasets'
PRETRAINED_WEIGHTS = '/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/pretrain/ir50.pth'
RESULTS_DIR = '/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/results_light_head'

# Mixup
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.15

# Augmentation
ROTATION_DEG = 10
TRANSLATE_FRAC = 0.06
COLOR_JITTER = 0.2

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Spatial Attention + Light Head
# ──────────────────────────────────────────────────────────────────────────────
#
# The original head (12.8M params) memorized spatial positions.
# Pure GAP (~530K) threw away ALL spatial info — but spatial info matters for
# emotions (mouth region for happiness, eye region for surprise, etc.).
#
# SOLUTION: Spatial Attention before GAP
#   Conv1x1(512→512) → Sigmoid → element-wise multiply → GAP → FC head
#   This learns WHICH spatial regions matter (~262K attention params)
#   while GAP keeps the model spatially invariant overall.


class SpatialAttention(nn.Module):
    """Channel-wise spatial attention: learns which 7×7 regions matter."""
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),  # 262K params
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.attn(x)


def replace_head_with_light(model, num_classes, drop_ratio=0.2):
    """Replace the heavy FC head with attention + GAP head (~795K params).
    
    Architecture:
      SpatialAttention → BN → GAP → FC(512→512) → FC(512→512) → FC(512→6)
    """
    model.output_layer = nn.Sequential(
        SpatialAttention(512),       # learns which 7×7 regions matter (~262K)
        BatchNorm2d(512),
        AdaptiveAvgPool2d(1),        # 7×7×512 → 1×1×512
        Flatten(),
        Dropout(drop_ratio),         # 0.2
        nn.Linear(512, 512),         # 262K — remap identity features
        BatchNorm1d(512),
        nn.ReLU(inplace=True),
        Dropout(0.15),
        nn.Linear(512, 512),         # 262K — refine emotion features
        BatchNorm1d(512),
        nn.ReLU(inplace=True),
        Dropout(0.1),
        nn.Linear(512, num_classes), # 3K
    )

    # Xavier init for linear layers, Kaiming for conv
    for m in model.output_layer.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    total_head = sum(p.numel() for p in model.output_layer.parameters())
    print(f"  Attention + Light head installed: {total_head:,} params (was ~12.8M)")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Loss: Class-Balanced Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Mixup helpers
# ──────────────────────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────────────
# Data Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms():
    train_transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([
            v2.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER,
                           saturation=0.15, hue=0.03)
        ], p=0.5),
        v2.RandomAffine(degrees=ROTATION_DEG, translate=(TRANSLATE_FRAC, TRANSLATE_FRAC),
                        scale=(0.95, 1.05), shear=3),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
        v2.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0.0),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, val_transform


def count_samples(dataset):
    counts = [0] * len(CLASSES)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        ds_list = dataset.datasets
    else:
        ds_list = [dataset]
    for ds in ds_list:
        if hasattr(ds, 'targets'):
            targets = ds.targets
        elif hasattr(ds, 'samples'):
            targets = [s[1] for s in ds.samples]
        else:
            continue
        for t in targets:
            if 0 <= t < len(CLASSES):
                counts[t] += 1
    return counts


def load_data():
    train_tfm, val_tfm = get_transforms()

    train_datasets = []
    val_datasets = []

    if os.path.isdir(DATASET_ROOT):
        for ds_name in sorted(os.listdir(DATASET_ROOT)):
            ds_path = os.path.join(DATASET_ROOT, ds_name)
            if not os.path.isdir(ds_path):
                continue

            # Training: 'train' + 'eval' splits
            for sub in ['train', 'eval']:
                p = os.path.join(ds_path, sub)
                if os.path.isdir(p):
                    try:
                        ds = torchvision.datasets.ImageFolder(p, transform=train_tfm)
                        train_datasets.append(ds)
                        print(f"  [TRAIN] {ds_name}/{sub}: {len(ds)} samples")
                    except Exception:
                        pass

            # Validation: 'test' split from same datasets
            test_p = os.path.join(ds_path, 'test')
            if os.path.isdir(test_p):
                try:
                    ds = torchvision.datasets.ImageFolder(test_p, transform=val_tfm)
                    val_datasets.append(ds)
                    print(f"  [VAL]   {ds_name}/test:  {len(ds)} samples")
                except Exception:
                    pass

    if not train_datasets:
        print("No training datasets found!")
        sys.exit(1)

    full_train_ds = torch.utils.data.ConcatDataset(train_datasets)
    full_val_ds = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None
    samples_per_cls = count_samples(full_train_ds)

    print(f"\n  Total train samples: {len(full_train_ds):,}")
    if full_val_ds:
        print(f"  Total val samples:   {len(full_val_ds):,}")

    train_loader = torch.utils.data.DataLoader(
        full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = None
    if full_val_ds:
        val_loader = torch.utils.data.DataLoader(
            full_val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
    return train_loader, val_loader, samples_per_cls


# ──────────────────────────────────────────────────────────────────────────────
# Layer Freezing (same 2 options as train_ir50_layer_freezing.py)
# ──────────────────────────────────────────────────────────────────────────────
def freeze_layers(model):
    """Freeze input_layer + body1-3. Unfreeze body4 + light head."""
    frozen_modules = [model.input_layer, model.body1, model.body2, model.body3]
    frozen_names = ['input_layer', 'body1', 'body2', 'body3']

    total_frozen = 0
    for module in frozen_modules:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
            total_frozen += param.numel()

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    body4_params = sum(p.numel() for p in model.body4.parameters())
    head_params = sum(p.numel() for p in model.output_layer.parameters())

    print(f"\n{'='*60}")
    print(f"  FREEZE: {', '.join(frozen_names)}")
    print(f"  TRAIN:  body4 ({body4_params:,}) + light_head ({head_params:,})")
    print(f"  Frozen params : {total_frozen:,}")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"{'='*60}\n")

    return frozen_modules


def get_param_groups(model, backbone_lr, head_lr):
    """Differential LR: body4 → backbone_lr (gentle), head → head_lr (fast)."""
    head_params = list(model.output_layer.parameters())
    head_ids = set(id(p) for p in head_params)

    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]

    groups = []
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': backbone_lr, 'name': 'body4'})
    groups.append({'params': head_params, 'lr': head_lr, 'name': 'head'})

    for g in groups:
        n = sum(p.numel() for p in g['params'] if p.requires_grad)
        print(f"  Param group '{g['name']}': lr={g['lr']:.6f}, params={n:,}")

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, 'ir50_light_head_results.csv')
    best_ckpt = os.path.join(RESULTS_DIR, 'ir50_light_head_best.pth')
    last_ckpt = os.path.join(RESULTS_DIR, 'ir50_light_head_last.pth')

    # ── Model ────────────────────────────────────────────────────────────────
    print("Initializing IR50 with Light Head...")
    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se', num_classes=len(CLASSES))

    # Load CelebA pretrained weights (into the ORIGINAL heavy head first)
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading CelebA pretrained weights from {PRETRAINED_WEIGHTS}")
        load_pretrained_weights(model, torch.load(PRETRAINED_WEIGHTS, map_location='cpu'))
    else:
        print(f"WARNING: Pretrained weights not found at {PRETRAINED_WEIGHTS}")

    # NOW replace the head with the light version (discards the old head weights)
    model = replace_head_with_light(model, num_classes=len(CLASSES), drop_ratio=0.5)

    # Apply layer freezing (freeze input_layer + body1-3)
    frozen_modules = freeze_layers(model)
    model = model.to(device)

    # EMA
    model_ema = ModelEmaV2(model, decay=0.999)

    # ── Optimizer with differential LR ──────────────────────────────────────
    param_groups = get_param_groups(model, backbone_lr=LR, head_lr=HEAD_LR)
    optimizer = torch.optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, lr_min=1e-5,
        warmup_t=5, warmup_lr_init=1e-5, warmup_prefix=True,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, samples_per_cls = load_data()
    print(f"Samples per class: {samples_per_cls}")

    train_criterion = ClassBalancedFocalLoss(samples_per_cls).to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'ema_state_dict' in ckpt:
            model_ema.module.load_state_dict(ckpt['ema_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed at epoch {start_epoch + 1}, best_acc={best_acc:.2f}%")
        freeze_layers(model)
    elif args.resume:
        print(f"WARNING: Resume path {args.resume} not found. Starting from scratch.")

    # ── CSV setup ───────────────────────────────────────────────────────────
    if start_epoch == 0 or not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "Epoch", "LR", "Train Loss", "Train Acc", "Val Loss", "Val Acc"
            ])

    # ── Training Loop ───────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs (from epoch {start_epoch + 1})...")
    print(f"Using device: {device}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for m in frozen_modules:
            m.eval()

        run_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            do_mixup = np.random.rand() < MIXUP_PROB
            if do_mixup:
                inputs, y_a, y_b, lam = mixup_batch(inputs, targets, alpha=MIXUP_ALPHA)

            outputs = model(inputs)

            if do_mixup:
                loss = mixup_loss(train_criterion, outputs, y_a, y_b, lam)
            else:
                loss = train_criterion(outputs, targets)

            loss = loss / ACCUM_STEPS
            loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                model_ema.update(model)

            run_loss += loss.item() * ACCUM_STEPS * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)

            pbar.set_postfix({'loss': f'{run_loss / total:.4f}', 'acc': f'{100. * correct / total:.2f}'})

        scheduler.step(epoch)

        train_loss = run_loss / total
        train_acc = 100. * correct / total

        # ── Validation ──────────────────────────────────────────────────────
        val_acc = 0.0
        val_loss_avg = 0.0

        if val_loader:
            model_ema.module.eval()
            val_correct = 0
            val_total = 0
            val_loss_sum = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model_ema.module(inputs)
                    loss = val_criterion(outputs, targets)
                    val_loss_sum += loss.item() * inputs.size(0)
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(targets).sum().item()
                    val_total += inputs.size(0)

            val_acc = 100. * val_correct / val_total
            val_loss_avg = val_loss_sum / val_total
            print(f"Validation Acc: {val_acc:.2f}% | Loss: {val_loss_avg:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model_ema.module.state_dict(), best_ckpt)
                print(f"  ★ New Best Saved! ({best_acc:.2f}%) → {best_ckpt}")

        # ── Save last checkpoint ────────────────────────────────────────────
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': model_ema.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, last_ckpt)

        # ── CSV log ─────────────────────────────────────────────────────────
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                optimizer.param_groups[0]['lr'],
                train_loss,
                train_acc,
                val_loss_avg,
                val_acc,
            ])

        print(f"  Epoch {epoch + 1} done | Train: {train_acc:.2f}% | "
              f"Val: {val_acc:.2f}% | Best: {best_acc:.2f}%\n")

    print(f"\nTraining complete! Best val accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {RESULTS_DIR}")

    # ── Final Test on external datasets (CK+, KDEF) ────────────────────────
    if os.path.isdir(TEST_DATASET_ROOT):
        print(f"\n{'='*60}")
        print(f"  FINAL TEST on external datasets ({TEST_DATASET_ROOT})")
        print(f"{'='*60}")

        # Load best model
        if os.path.exists(best_ckpt):
            best_state = torch.load(best_ckpt, map_location=device)
            model_ema.module.load_state_dict(best_state)
            print(f"  Loaded best checkpoint (val acc: {best_acc:.2f}%)\n")
        else:
            print("  WARNING: Best checkpoint not found, using last model.\n")

        model_ema.module.eval()
        _, val_tfm = get_transforms()

        for ds_name in sorted(os.listdir(TEST_DATASET_ROOT)):
            ds_path = os.path.join(TEST_DATASET_ROOT, ds_name)
            if not os.path.isdir(ds_path):
                continue

            # Handle both structures:
            #   CKplusIm/  → has train/eval/test subfolders → load ALL splits
            #   KDEF/      → has class folders directly → use root
            test_sub = os.path.join(ds_path, 'test')
            if os.path.isdir(test_sub):
                # Dataset has splits — concatenate all of them
                split_datasets = []
                for split in ['train', 'eval', 'test']:
                    sp = os.path.join(ds_path, split)
                    if os.path.isdir(sp):
                        try:
                            split_datasets.append(torchvision.datasets.ImageFolder(sp, transform=val_tfm))
                        except Exception:
                            pass
                if not split_datasets:
                    continue
                test_ds = torch.utils.data.ConcatDataset(split_datasets)
            else:
                # Class folders directly
                try:
                    test_ds = torchvision.datasets.ImageFolder(ds_path, transform=val_tfm)
                except Exception:
                    continue

            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
            )

            test_correct = 0
            test_total = 0
            test_loss_sum = 0.0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model_ema.module(inputs)
                    loss = val_criterion(outputs, targets)
                    test_loss_sum += loss.item() * inputs.size(0)
                    _, preds = outputs.max(1)
                    test_correct += preds.eq(targets).sum().item()
                    test_total += targets.size(0)

            test_acc = 100. * test_correct / test_total if test_total > 0 else 0.0
            test_loss = test_loss_sum / test_total if test_total > 0 else 0.0
            print(f"  [TEST] {ds_name}: Acc={test_acc:.2f}% | Loss={test_loss:.4f} | Samples={test_total}")

        print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='IR50 Transfer Learning with Light Head (freeze body1-3, train body4+head)',
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"{'='*60}")
    print(f"  IR50 Transfer Learning — Light Head")
    print(f"  Strategy: freeze body1-3, fine-tune body4 (lr={LR}) + head (lr={HEAD_LR})")
    print(f"  Epochs: {args.epochs}")
    print(f"  Resume: {args.resume or 'None'}")
    print(f"{'='*60}")
    train(args)
