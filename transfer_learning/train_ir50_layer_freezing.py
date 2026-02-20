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
from tqdm import tqdm
from timm.utils import ModelEmaV2
from timm.scheduler import CosineLRScheduler

# Import IR50
try:
    from ir50 import Backbone, load_pretrained_weights
except ImportError:
    print("Error: Could not import ir50.py. Ensure it is in the same directory.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (best practices carried over from train_ir50.py)
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
ACCUM_STEPS = 2
EPOCHS = 25
LR = 0.005                # Lower than full-finetune (0.02) – fewer trainable params need gentler LR
HEAD_LR_MULT = 2.0        # Classification head learns 2x faster (random init)
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
IMG_SIZE = 112             # IR50 native resolution
NUM_WORKERS = 8
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Paths
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets'
TEST_DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/test_datasets'
PRETRAINED_WEIGHTS = '/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/pretrain/ir50.pth'
RESULTS_DIR = '/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/results_freeze_layers' #chage if needed

# Mixup (matched from train_ir50.py)
MIXUP_ALPHA = 0.15
MIXUP_PROB = 0.3

# Augmentation intensities (matched from train_ir50.py)
ROTATION_DEG = 10
TRANSLATE_FRAC = 0.06
COLOR_JITTER = 0.2
GRAYSCALE_PROB = 0.3

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Loss: Class-Balanced Focal Loss (from train_ir50.py)
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
# Mixup helpers (from train_ir50.py)
# ──────────────────────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────────────
# Data Pipeline (from train_ir50.py)
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

    # Train Data
    train_datasets = []
    if os.path.isdir(DATASET_ROOT):
        for ds_name in os.listdir(DATASET_ROOT):
            ds_path = os.path.join(DATASET_ROOT, ds_name)
            if not os.path.isdir(ds_path):
                continue
            for sub in ['train', 'eval']:
                p = os.path.join(ds_path, sub)
                if os.path.isdir(p):
                    try:
                        train_datasets.append(torchvision.datasets.ImageFolder(p, transform=train_tfm))
                    except Exception:
                        pass

    if not train_datasets:
        print("No training datasets found!")
        sys.exit(1)

    full_train_ds = torch.utils.data.ConcatDataset(train_datasets)
    samples_per_cls = count_samples(full_train_ds)

    # Val Data
    val_datasets = []
    if os.path.isdir(TEST_DATASET_ROOT):
        for ds_name in os.listdir(TEST_DATASET_ROOT):
            ds_path = os.path.join(TEST_DATASET_ROOT, ds_name)
            if not os.path.isdir(ds_path):
                continue
            try:
                ds = torchvision.datasets.ImageFolder(ds_path, transform=val_tfm)
                val_datasets.append(ds)
            except Exception:
                pass

    full_val_ds = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None

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
# Layer Freezing
# ──────────────────────────────────────────────────────────────────────────────
#  Option 1 (--freeze_option 1):
#    Freeze: input_layer + body1 + body2 + body3          (~9M params frozen)
#    Train:  body4 + output_layer                         (~17.8M params trainable)
#
#  Option 2 (--freeze_option 2):
#    Freeze: input_layer + body1 + body2 + body3 + body4  (~14M params frozen)
#    Train:  output_layer only                            (~12.8M params trainable)
#

def freeze_layers(model, option):
    """Freeze layers based on the selected option.
    
    Frozen layers have requires_grad=False and are set to eval() mode
    so that BatchNorm running statistics stay frozen (CelebA stats).
    """
    if option == 1:
        frozen_modules = [model.input_layer, model.body1, model.body2, model.body3]
        frozen_names = ['input_layer', 'body1', 'body2', 'body3']
    elif option == 2:
        frozen_modules = [model.input_layer, model.body1, model.body2, model.body3, model.body4]
        frozen_names = ['input_layer', 'body1', 'body2', 'body3', 'body4']
    else:
        raise ValueError(f"Invalid freeze option: {option}. Must be 1 or 2.")

    total_frozen = 0
    for module in frozen_modules:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
            total_frozen += param.numel()

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"  FREEZE OPTION {option}")
    print(f"  Frozen layers : {', '.join(frozen_names)}")
    print(f"  Frozen params : {total_frozen:,}")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"{'='*60}\n")

    return frozen_modules


def get_param_groups(model, option, base_lr):
    """Create parameter groups with differential learning rates.
    
    - Unfrozen backbone layers (body4 in option 1) → base_lr
    - Classification head (output_layer) → base_lr * HEAD_LR_MULT
    """
    head_params = list(model.output_layer.parameters())
    head_param_ids = set(id(p) for p in head_params)

    # All trainable params that are NOT in the head
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_param_ids]

    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr,
            'name': 'backbone_unfrozen',
        })
    param_groups.append({
        'params': head_params,
        'lr': base_lr * HEAD_LR_MULT,
        'name': 'head',
    })

    for g in param_groups:
        n_params = sum(p.numel() for p in g['params'] if p.requires_grad)
        print(f"  Param group '{g['name']}': lr={g['lr']:.6f}, params={n_params:,}")

    return param_groups


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, f'ir50_freeze_option{args.freeze_option}_results.csv')
    best_ckpt_path = os.path.join(RESULTS_DIR, f'ir50_freeze_option{args.freeze_option}_best.pth')
    last_ckpt_path = os.path.join(RESULTS_DIR, f'ir50_freeze_option{args.freeze_option}_last.pth')

    # ── Model ────────────────────────────────────────────────────────────────
    print("Initializing Model...")
    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se', num_classes=len(CLASSES))

    # Load CelebA pretrained weights
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading CelebA pretrained weights from {PRETRAINED_WEIGHTS}")
        load_pretrained_weights(model, torch.load(PRETRAINED_WEIGHTS, map_location='cpu'))
    else:
        print(f"WARNING: Pretrained weights not found at {PRETRAINED_WEIGHTS}")

    # Apply layer freezing
    frozen_modules = freeze_layers(model, args.freeze_option)
    model = model.to(device)

    # EMA (from train_ir50.py)
    model_ema = ModelEmaV2(model, decay=0.999)

    # ── Optimizer with differential LR ──────────────────────────────────────
    param_groups = get_param_groups(model, args.freeze_option, LR)
    optimizer = torch.optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Cosine LR scheduler with warmup (from train_ir50.py)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, lr_min=1e-5,
        warmup_t=5, warmup_lr_init=1e-4, warmup_prefix=True,
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

        # Re-apply freezing after loading state (to ensure frozen layers stay frozen)
        freeze_layers(model, args.freeze_option)
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
        # Set model to train, but keep frozen layers in eval mode
        model.train()
        for m in frozen_modules:
            m.eval()

        run_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixup (from train_ir50.py)
            do_mixup = np.random.rand() < MIXUP_PROB
            if do_mixup:
                inputs, y_a, y_b, lam = mixup_batch(inputs, targets, alpha=MIXUP_ALPHA)

            # Forward
            outputs = model(inputs)

            if do_mixup:
                loss = mixup_loss(train_criterion, outputs, y_a, y_b, lam)
            else:
                loss = train_criterion(outputs, targets)

            # Gradient accumulation (from train_ir50.py)
            loss = loss / ACCUM_STEPS
            loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                # Gradient clipping (from train_ir50.py)
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

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model_ema.module.state_dict(), best_ckpt_path)
                print(f"  ★ New Best Saved! ({best_acc:.2f}%) → {best_ckpt_path}")

        # ── Save last checkpoint (for resume) ───────────────────────────────
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': model_ema.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'freeze_option': args.freeze_option,
        }, last_ckpt_path)

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


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='IR50 Transfer Learning with Layer Freezing',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--freeze_option', type=int, default=1, choices=[1, 2],
        help=(
            'Which freeze strategy to use:\n'
            '  1 = Freeze input_layer+body1+body2+body3, train body4+head (~17.8M trainable)\n'
            '  2 = Freeze entire backbone, train head only (~12.8M trainable)\n'
            '(default: 1)'
        ),
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from (e.g. results_freeze_layers/ir50_freeze_option1_last.pth)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"{'='*60}")
    print(f"  IR50 Transfer Learning — Layer Freezing")
    print(f"  Freeze Option: {args.freeze_option}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Resume: {args.resume or 'None'}")
    print(f"{'='*60}")
    train(args)
