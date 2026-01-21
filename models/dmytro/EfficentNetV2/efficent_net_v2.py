
import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2, accuracy
from timm.scheduler import CosineLRScheduler

# Ensure src is importable if needed (legacy check)
if os.path.exists('src') and not os.path.islink('src'):
    pass 
else:
    project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    compact_transformers_path = os.path.join(project_root, 'Compact-Transformers')
    if compact_transformers_path not in sys.path:
        sys.path.append(compact_transformers_path)

# --- Configuration ---
BATCH_SIZE = 128         # High Batch Size as recommended
ACCUM_STEPS = 1          # Adjusted based on VRAM (128 usually fits S models, if not, increase this)
EPOCHS = 600
LR = 0.001 * (BATCH_SIZE / 256) # Scale LR
WEIGHT_DECAY = 0.05
IMG_SIZE = 64
NUM_WORKERS = 8
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_1_0_ready_to_use_datasets'
DATASETS = ['AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB', 'KDEFFormated',\
     'jaffeFormated', 'MMAFEDB', 'NONAMEFormated', 'ExpWFormated', 'EmoSet-118k']

# --- Overfitting Fix Hyperparameters ---
DROP_PATH_RATE = 0.2     # Linear profile handled by timm
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
LABEL_SMOOTHING = 0.1
RANDAUG_N = 2
RANDAUG_M = 9
WARMUP_EPOCHS = 5        # Per recipe (or 20 if matching Vim, sticking to 5 per prompt text)

# --- Dataset Statistics (Calculated from 114k set) ---
MEAN = [0.4681, 0.4447, 0.4560]
STD = [0.2327, 0.2227, 0.2224]

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ## 1. Data Pipeline

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(num_ops=RANDAUG_N, magnitude=RANDAUG_M),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_datasets = []
val_datasets = []
emotion_counts_train = {c: 0 for c in CLASSES}

print("Loading datasets...")
for ds_name in DATASETS:
    ds_path = os.path.join(DATASET_ROOT, ds_name)
    train_path = os.path.join(ds_path, 'train')
    eval_path = os.path.join(ds_path, 'eval')
    
    if os.path.isdir(train_path):
        ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        train_datasets.append(ds)
        # Faster counting
        if hasattr(ds, 'targets'):
            counts = Counter(ds.targets)
            for idx, count in counts.items():
                class_name = ds.classes[idx]
                if class_name in emotion_counts_train:
                    emotion_counts_train[class_name] += count
        else:
            for _, label_idx in ds.samples:
                class_name = ds.classes[label_idx]
                if class_name in emotion_counts_train:
                    emotion_counts_train[class_name] += 1
    
    if os.path.isdir(eval_path):
        ds_val = torchvision.datasets.ImageFolder(root=eval_path, transform=transform_val)
        val_datasets.append(ds_val)

if not train_datasets:
    print("Error: No datasets found.")
    sys.exit(1)

full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
full_val_dataset = torch.utils.data.ConcatDataset(val_datasets)

train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Total Training Samples: {len(full_train_dataset)}")
print(f"Total Validation Samples: {len(full_val_dataset)}")

# --- Mixup Fn ---
mixup_fn = Mixup(
    mixup_alpha=MIXUP_ALPHA, 
    cutmix_alpha=CUTMIX_ALPHA, 
    prob=1.0, 
    switch_prob=0.5, 
    mode='batch',
    label_smoothing=LABEL_SMOOTHING, 
    num_classes=len(CLASSES)
)

# ## 2. Model Initialization (EfficientNetV2-S with Stem Fix)

def create_cifar_efficientnet():
    print("Creating EfficientNetV2-S with Overfitting Fix...")
    model = timm.create_model(
        'tf_efficientnetv2_s', 
        pretrained=False, 
        num_classes=len(CLASSES), 
        drop_path_rate=DROP_PATH_RATE
    )
    
    # 1. Modify Stem Stride (The "64x64 Fix")
    # Original is stride=2. We force stride=1 to keep 64x64 resolution at start.
    old_stem = model.conv_stem
    model.conv_stem = nn.Conv2d(
        in_channels=old_stem.in_channels,
        out_channels=old_stem.out_channels,
        kernel_size=old_stem.kernel_size,
        stride=(1, 1), # CRITICAL FIX
        padding=old_stem.padding,
        bias=False
    )
    # Re-init stem
    nn.init.kaiming_normal_(model.conv_stem.weight, mode='fan_out', nonlinearity='relu')
    print("  -> Modified Stem Stride to 1")

    # 2. Modify Stage 2 Stride (Optional but Recommended)
    # Typically block 1 or 2. In EfficientNetV2, blocks are in model.blocks
    # We look for the first block that downsamples (stride=2) and set it to 1 if it's early.
    # Stage 0 is stem. Stage 1 is blocks[0]. Stage 2 is usually blocks[1] or [2].
    # Let's inspect block strides.
    # For safety/simplicity in this script without deep inspection loop, we stick to the Stem Fix 
    # which ensures 64x64 enters the first stage (vs 32x32).
    # If more aggressive resolution preservation is needed, we would modify `model.blocks[1][0].conv_dw.stride`.
    
    return model

model = create_cifar_efficientnet()
model = model.to(device)

# EMA
model_ema = ModelEmaV2(model, decay=0.9999)

# ## 3. Optimization

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8, betas=(0.9, 0.999))

scheduler = CosineLRScheduler(
    optimizer,
    t_initial=EPOCHS,
    lr_min=1e-5,
    warmup_t=WARMUP_EPOCHS,
    warmup_lr_init=1e-6,
    warmup_prefix=True
)

train_criterion = SoftTargetCrossEntropy()
val_criterion = torch.nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_val_acc = 0.0

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        # AMP
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = train_criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = train_criterion(outputs, targets)
        
        # Accumulation check
        loss = loss / ACCUM_STEPS
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model_ema.update(model)
        
        # Stats
        loss_val = loss.item() * ACCUM_STEPS # Scale back for logging
        running_loss += loss_val * inputs.size(0)
        
        # Acc
        if targets.ndim == 2:
            preds = outputs.argmax(dim=1)
            labels = targets.argmax(dim=1)
        else:
            preds = outputs.argmax(dim=1)
            labels = targets
        correct += preds.eq(labels).sum().item()
        total += inputs.size(0)
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': running_loss / total, 'acc': 100. * correct / total, 'lr': current_lr})
        
    scheduler.step(epoch)
    train_acc = 100. * correct / total
    train_loss = running_loss / total
    print(f"Train     - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    return train_loss

def validate(epoch):
    model_ema.module.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model_ema.module(inputs)
            loss = val_criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    loss_avg = running_loss / total
    print(f"Validation (EMA) - Loss: {loss_avg:.4f}, Acc: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run a short test')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint (efficientnetv2_last.pth)')
    args = parser.parse_args()

    if args.dry_run:
        print("Running Dry Run...")
        if len(full_train_dataset) > 0:
            train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=2, shuffle=True)
            inputs, targets = next(iter(train_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            if mixup_fn: inputs, targets = mixup_fn(inputs, targets)
            
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = train_criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = train_criterion(outputs, targets)
            loss.backward()
            print("Dry Run Successful!")
        sys.exit(0)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, Best Acc: {best_val_acc:.2f}%")

    save_dir = '/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/EfficentNetV2/'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(epoch)
        val_acc = validate(epoch)
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_ema_state_dict': model_ema.module.state_dict(),
                'best_acc': best_val_acc
            }, os.path.join(save_dir, 'efficientnetv2_best.pth'))
            print(f"Saved Best Model: {best_val_acc:.2f}%")
            
        # Save Last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'scheduler_state_dict': scheduler.state_dict(),
            'model_ema_state_dict': model_ema.module.state_dict(),
            'best_acc': best_val_acc
        }, os.path.join(save_dir, 'efficientnetv2_last.pth'))
