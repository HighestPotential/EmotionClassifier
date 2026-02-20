import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import numpy as np
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
BATCH_SIZE = 128     
ACCUM_STEPS = 1          
EPOCHS = 400
LR = 0.002 * (BATCH_SIZE / 256) 
WEIGHT_DECAY = 0.05
IMG_SIZE = 64
NUM_WORKERS = 8
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets'
DATASETS = ['AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB', 'KDEFFormated',\
     'jaffeFormated', 'MMAFEDB', 'ExpWFormated', 'EmoSet-118k', 'NHFI']

# --- Overfitting Fix Hyperparameters ---
DROP_PATH_RATE = 0.3
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 0.2
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5

# --- CB-FL Specific: Mixup probability reduced (Option C) ---
MIXUP_PROB = 0.3  # Only 30% of batches get Mixup; 70% use CB-FL with hard targets

# --- CB-FL Hyperparameters ---
CBFL_BETA = 0.9999
CBFL_GAMMA = 2.0

# --- Augmentation Intensity Settings ---
ROTATION_DEG = 15
TRANSLATE_FRAC = 0.05
COLOR_JITTER = 0.2
SHARPNESS_FACTOR = 2.0
GRAYSCALE_PROB = 0.3

# --- Dataset Statistics ---
MEAN = [0.4681, 0.4447, 0.4560]
STD = [0.2327, 0.2227, 0.2224]

# =====================================================================
# CLASS-BALANCED FOCAL LOSS (Cui et al., CVPR 2019 + Lin et al., 2017)
# =====================================================================
class ClassBalancedFocalLoss(nn.Module):
    """
    Combines two ideas:
    1. Class-Balanced weighting: uses 'effective number' of samples
       E_n = (1 - beta^n) / (1 - beta) to compute per-class weights.
       This models diminishing returns of adding more data to large classes.
    2. Focal modulation: (1 - p_t)^gamma down-weights easy/confident predictions,
       forcing the model to focus on hard misclassifications (typically minority classes).
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        samples = torch.tensor(samples_per_class, dtype=torch.float32)
        
        # Effective number weighting
        effective_num = 1.0 - torch.pow(beta, samples)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples)  # Normalize so sum = num_classes

        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        # Standard cross-entropy with class weights (per-sample)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            reduction="none"
        )
        # Focal modulation
        pt = torch.exp(-ce_loss)           # p_t = probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# Set seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ## 1. Data Pipeline

transform_raw = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_datasets = []
val_datasets = []
val_datasets_map = {}
emotion_counts_train = {c: 0 for c in CLASSES}

print("Loading datasets...")
for ds_name in DATASETS:
    ds_path = os.path.join(DATASET_ROOT, ds_name)
    train_path = os.path.join(ds_path, 'train')
    eval_path = os.path.join(ds_path, 'eval')
    test_path = os.path.join(ds_path, 'test')
    
    if os.path.isdir(train_path):
        ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform_raw)
        train_datasets.append(ds)
        if hasattr(ds, 'targets'):
            counts = Counter(ds.targets)
            for idx, count in counts.items():
                cls = ds.classes[idx]
                if cls in emotion_counts_train: emotion_counts_train[cls] += count
        else:
             for _, label_idx in ds.samples:
                cls = ds.classes[label_idx]
                if cls in emotion_counts_train: emotion_counts_train[cls] += 1
    
    if os.path.isdir(eval_path):
        ds_val = torchvision.datasets.ImageFolder(root=eval_path, transform=transform_raw)
        val_datasets.append(ds_val)
        val_datasets_map[ds_name] = ds_val

    if os.path.isdir(test_path):
        ds_test = torchvision.datasets.ImageFolder(root=test_path, transform=transform_raw)
        train_datasets.append(ds_test)
        if hasattr(ds_test, 'targets'):
            counts = Counter(ds_test.targets)
            for idx, count in counts.items():
                cls = ds_test.classes[idx]
                if cls in emotion_counts_train: emotion_counts_train[cls] += count
        else:
             for _, label_idx in ds_test.samples:
                cls = ds_test.classes[label_idx]
                if cls in emotion_counts_train: emotion_counts_train[cls] += 1

if not train_datasets:
    print("Error: No datasets found.")
    sys.exit(1)

full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
full_val_dataset = torch.utils.data.ConcatDataset(val_datasets)

# --- Dynamic Mean/Std Calculation ---
print("Calculating mean and std of the training dataset...")
loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

mean = 0.
std = 0.
nb_samples = 0.

for data, _ in tqdm(loader, desc="Computing Stats"):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

MEAN = mean.tolist()
STD = std.tolist()
print(f"Calculated Mean: {MEAN}")
print(f"Calculated Std:  {STD}")

# --- Final Transforms ---
transform_train_final = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.Pad(padding=8, padding_mode='reflect'),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomApply([
        v2.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=3)
    ], p=0.5),
    v2.CenterCrop(IMG_SIZE),
    v2.RandomApply([
        v2.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=COLOR_JITTER)
    ], p=0.4),
    v2.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=SHARPNESS_FACTOR)], p=0.2),
    v2.RandomApply([v2.RandomAutocontrast()], p=0.2),
    v2.RandomGrayscale(p=GRAYSCALE_PROB),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
    v2.RandomErasing(p=0.1, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0.0),
])

transform_val_final = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

print("Updating transforms...")
for ds in train_datasets:
    ds.transform = transform_train_final
for ds in val_datasets:
    ds.transform = transform_val_final

# --- Class Balancing (WeightedRandomSampler) ---
print("Calculating class weights for balancing...")
all_targets = []
for ds in train_datasets:
    if hasattr(ds, 'targets'):
        all_targets.extend(ds.targets)
    else:
        for _, label_idx in ds.samples:
            all_targets.append(label_idx)

all_targets = np.array(all_targets)
total_samples = len(all_targets)
class_sample_counts = np.bincount(all_targets)
class_weights = total_samples / (class_sample_counts + 1e-6)

print(f"Class Sample Counts: {class_sample_counts}")
print(f"Class Weights: {class_weights}")

samples_weights = class_weights[all_targets]
samples_weights = torch.from_numpy(samples_weights).double()

sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True, sampler=sampler, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

target_val_loaders = {}
for name, ds in val_datasets_map.items():
    target_val_loaders[name] = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Total Training Samples: {len(full_train_dataset)}")
print(f"Total Validation Samples: {len(full_val_dataset)}")

# --- Mixup Fn (Reduced probability for Option C) ---
mixup_fn = Mixup(
    mixup_alpha=MIXUP_ALPHA, 
    cutmix_alpha=CUTMIX_ALPHA,
    prob=MIXUP_PROB,            # Only 30% of batches get Mixup/CutMix
    switch_prob=0.3,
    mode='batch',
    label_smoothing=LABEL_SMOOTHING, 
    num_classes=len(CLASSES)
)

# ## 2. Model

def create_cifar_efficientnet():
    print("Creating EfficientNetV2-S (V4: CB-FL + AdamW)...")
    model = timm.create_model(
        'tf_efficientnetv2_s', 
        pretrained=False, 
        num_classes=len(CLASSES), 
        drop_path_rate=DROP_PATH_RATE
    )
    
    old_stem = model.conv_stem
    model.conv_stem = nn.Conv2d(
        in_channels=old_stem.in_channels,
        out_channels=old_stem.out_channels,
        kernel_size=old_stem.kernel_size,
        stride=(1, 1),
        padding=old_stem.padding,
        bias=False
    )
    nn.init.kaiming_normal_(model.conv_stem.weight, mode='fan_out', nonlinearity='relu')
    print("  -> Modified Stem Stride to 1")
    return model

model = create_cifar_efficientnet()
model = model.to(device)

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

# --- Loss Functions ---
# SoftTargetCrossEntropy for Mixup batches (30%)
soft_criterion = SoftTargetCrossEntropy()
# CB-FL for non-Mixup batches (70%) — initialized after class counts are known
samples_per_cls = [int(class_sample_counts[i]) for i in range(len(CLASSES))]
print(f"Samples per class for CB-FL: {samples_per_cls}")
cbfl_criterion = ClassBalancedFocalLoss(samples_per_cls, beta=CBFL_BETA, gamma=CBFL_GAMMA).to(device)

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
        
        # Save original integer targets BEFORE Mixup
        original_targets = targets.clone()
        
        # Apply Mixup (prob=0.3, so 70% of the time targets stay as integers)
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        # Determine if Mixup was applied by checking target shape
        mixup_applied = (targets.ndim == 2)
        
        # AMP
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                if mixup_applied:
                    # Mixup was applied → use SoftTargetCrossEntropy
                    loss = soft_criterion(outputs, targets)
                else:
                    # No Mixup → use CB-FL with hard integer targets
                    loss = cbfl_criterion(outputs, original_targets)
        else:
            outputs = model(inputs)
            if mixup_applied:
                loss = soft_criterion(outputs, targets)
            else:
                loss = cbfl_criterion(outputs, original_targets)
        
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
        loss_val = loss.item() * ACCUM_STEPS
        running_loss += loss_val * inputs.size(0)
        
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

def validate(epoch, loader, name="Global"):
    model_ema.module.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_ema.module(inputs)
            loss = val_criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    loss_avg = running_loss / total
    print(f"Validation ({name}) - Loss: {loss_avg:.4f}, Acc: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run a short test')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()

    if args.dry_run:
        print("Running Dry Run...")
        if len(full_train_dataset) > 0:
            train_loader_dry = torch.utils.data.DataLoader(full_train_dataset, batch_size=2, shuffle=True)
            inputs, targets = next(iter(train_loader_dry))
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Test CB-FL path
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = cbfl_criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = cbfl_criterion(outputs, targets)
            loss.backward()
            print("CB-FL path: OK")
            
            optimizer.zero_grad()
            
            # Test Mixup + SoftTarget path
            if mixup_fn: inputs, targets = mixup_fn(inputs, targets)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = soft_criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = soft_criterion(outputs, targets)
            loss.backward()
            print("Mixup + SoftTarget path: OK")
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

    save_dir = '/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/EfficentNetV2_v4_AdamW/'
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, 'efficientnetv4_cbfl_results.csv')
    csv_header = ['Epoch', 'LR', 'Train Loss', 'Global Val Acc']
    dataset_names_sorted = sorted(target_val_loaders.keys())
    csv_header.extend([f'{name} Acc' for name in dataset_names_sorted])

    if start_epoch == 0 or not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(csv_header)
    
    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(epoch)
        
        val_acc = validate(epoch, val_loader, name="Global (Val)")
        
        per_dataset_accs = {}
        for name, loader in target_val_loaders.items():
            per_dataset_accs[name] = validate(epoch, loader, name=name)
        
        current_lr = optimizer.param_groups[0]['lr']
        csv_row = [epoch + 1, current_lr, f'{train_loss:.4f}', f'{val_acc:.2f}']
        csv_row.extend([f'{per_dataset_accs.get(name, 0.0):.2f}' for name in dataset_names_sorted])
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(csv_row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_ema_state_dict': model_ema.module.state_dict(),
                'best_acc': best_val_acc
            }, os.path.join(save_dir, 'efficientnetv4_best.pth'))
            print(f"Saved Best Model: {best_val_acc:.2f}%")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'scheduler_state_dict': scheduler.state_dict(),
            'model_ema_state_dict': model_ema.module.state_dict(),
            'best_acc': best_val_acc
        }, os.path.join(save_dir, 'efficientnetv4_last.pth'))
