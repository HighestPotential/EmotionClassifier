
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2, accuracy
from timm.scheduler import CosineLRScheduler

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
vim_root = os.path.abspath(os.path.join(current_dir, ".."))
if vim_root not in sys.path:
    sys.path.append(vim_root)

vim_package_dir = os.path.join(vim_root, "vim")
if vim_package_dir not in sys.path:
    sys.path.append(vim_package_dir)

try:
    from vim.models_mamba import VisionMamba
except ImportError:
    try:
        from models_mamba import VisionMamba
    except ImportError:
        print("Error importing VisionMamba. Checked paths:", sys.path)
        raise

# --- Configuration ---
BATCH_SIZE = 64
ACCUM_STEPS = 1
EPOCHS = 300
LR_MAX = 5e-4
WEIGHT_DECAY = 0.05
IMG_SIZE = 64
NUM_WORKERS = 8
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_1_0_ready_to_use_datasets'
DATASETS = ['AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB', 'KDEFFormated',\
     'jaffeFormated', 'MMAFEDB', 'NONAMEFormated', 'ExpWFormated', 'EmoSet-118k']
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# --- Regularization Hyperparameters ---
DROP_PATH_RATE = 0.1
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 1.0
RANDAUG_MAGNITUDE = 9
RANDAUG_NUM_OPS = 2
WARMUP_EPOCHS = 20

# --- Dataset Statistics ---
# Calculated from dataset
MEAN = [0.4681, 0.4447, 0.4560]
STD = [0.2327, 0.2227, 0.2224]

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ## 1. Data Loading Pipeline

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(num_ops=RANDAUG_NUM_OPS, magnitude=RANDAUG_MAGNITUDE),
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
        for _, label_idx in ds.samples:
            class_name = ds.classes[label_idx]
            if class_name in emotion_counts_train:
                emotion_counts_train[class_name] += 1
    
    if os.path.isdir(eval_path):
        ds_val = torchvision.datasets.ImageFolder(root=eval_path, transform=transform_val)
        val_datasets.append(ds_val)

if len(train_datasets) == 0:
    print(f"No datasets found in {DATASET_ROOT}. Please check path.")
    sys.exit(1)

full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
full_val_dataset = torch.utils.data.ConcatDataset(val_datasets)

train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader = torch.utils.data.DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Total Training Samples: {len(full_train_dataset)}")

# --- Mixup / Cutmix ---
mixup_fn = Mixup(
    mixup_alpha=MIXUP_ALPHA, 
    cutmix_alpha=CUTMIX_ALPHA, 
    prob=MIXUP_PROB, 
    switch_prob=0.5, 
    mode='batch',
    label_smoothing=LABEL_SMOOTHING, 
    num_classes=len(CLASSES)
)

# ## 2. Model Initialization (Vim-Tiny)

model = VisionMamba(
    img_size=IMG_SIZE,
    patch_size=8,
    stride=8,
    embed_dim=192,
    depth=24,
    channels=3,
    num_classes=len(CLASSES),
    drop_path_rate=DROP_PATH_RATE,
    use_middle_cls_token=True,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    final_pool_type='mean',
    if_abs_pos_embed=True,
    bimamba_type="v2",
)
model.to(device)

model_ema = ModelEmaV2(model, decay=0.9999)

# ## 3. Optimization Setup

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

# Scheduler: Linear Warmup -> Cosine Annealing
# Using CosineLRScheduler directly for better compatibility
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

scaler = torch.cuda.amp.GradScaler()

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
        
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = train_criterion(outputs, targets)
        else:
             outputs = model(inputs)
             loss = train_criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        model_ema.update(model)
        
        running_loss += loss.item() * inputs.size(0)
        
        # Accuracy calculation (handle Mixup soft targets)
        if targets.ndim == 2: # Soft targets
            preds = outputs.argmax(dim=1)
            labels = targets.argmax(dim=1)
        else: # Hard labels
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
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    if args.dry_run:
        print("Running Dry Run...")
        # Break dataset for speed
        if len(full_train_dataset) > 0:
            train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=2, shuffle=True)
            inputs, targets = next(iter(train_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            if mixup_fn:
                inputs, targets = mixup_fn(inputs, targets)
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = train_criterion(outputs, targets)
            else:
                 outputs = model(inputs)
                 loss = train_criterion(outputs, targets)
            loss.backward()
            print("Dry Run Successful!")
        else:
             print("Dry Run Failed: No Data.")
        sys.exit(0)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
             scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        # 1. Train
        train_loss = train_one_epoch(epoch)
        
        # 2. Validate (Only once, after updates)
        val_acc = validate(epoch)
        
        # 3. Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(vim_root, 'vim_tiny_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_ema_state_dict': model_ema.module.state_dict(),
                'best_acc': best_val_acc
            }, save_path)
            print(f"Saved Best Model: {best_val_acc:.2f}%")

        # 4. Save Last (Always, for resuming)
        last_path = os.path.join(vim_root, 'vim_tiny_last.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'scheduler_state_dict': scheduler.state_dict(),
            'model_ema_state_dict': model_ema.module.state_dict(),
            'best_acc': best_val_acc
        }, last_path)
