import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from sam import SAM, enable_running_stats, disable_running_stats

# Ensure imports work
try:
    from custom_cct import CCT
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from custom_cct import CCT

BATCH_SIZE = 16         
ACCUM_STEPS = 8
EPOCHS = 600
LR = 0.0005 * (BATCH_SIZE / 256) 
WEIGHT_DECAY = 0.05
IMG_SIZE = 64
NUM_WORKERS = 8
VALIDATE_FREQ = 5        # Validate every 5 epochs

DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_3_0_ready_to_use_datasets'
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

DATASETS = ['AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB', 'KDEFFormated',\
     'jaffeFormated', 'MMAFEDB', 'ExpWFormated', 'EmoSet-118k', 'NHFI']

ROTATION_DEG = 15       
TRANSLATE_FRAC = 0.05   
COLOR_JITTER = 0.2      
SHARPNESS_FACTOR = 2.0  
GRAYSCALE_PROB = 0.2    

transform_train = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    
    v2.Pad(padding=8, padding_mode='reflect'),
    
    v2.RandomChoice([
        v2.RandomRotation(degrees=ROTATION_DEG), 
        v2.RandomAffine(degrees=0, translate=(TRANSLATE_FRAC, TRANSLATE_FRAC)),
        v2.RandomHorizontalFlip(p=1.0), 
        v2.Identity()
    ]),
    
    v2.CenterCrop(IMG_SIZE),

    v2.RandomApply([
        v2.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=COLOR_JITTER)
    ], p=0.2),
    
    v2.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=SHARPNESS_FACTOR)], p=0.1),
    v2.RandomApply([v2.RandomAutocontrast()], p=0.1),
    v2.RandomGrayscale(p=0.1),
    
    # 7. Random Erasing (Occlusion robustness) - DISABLED
    # v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),

    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CCT Standard Norm [-1, 1]
])

transform_val = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_raw = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ## 1. Data Loading

train_datasets = []
val_datasets = []
emotion_counts_train = {c: 0 for c in CLASSES}

print("Loading Training datasets...")
for ds_name in DATASETS:
    ds_path = os.path.join(DATASET_ROOT, ds_name)
    train_path = os.path.join(ds_path, 'train')
    eval_path = os.path.join(ds_path, 'eval')
    test_path = os.path.join(ds_path, 'test') # Maximize Data
    
    if os.path.isdir(train_path):
        ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        train_datasets.append(ds)
    
    if os.path.isdir(eval_path):
        ds_val = torchvision.datasets.ImageFolder(root=eval_path, transform=transform_val)
        val_datasets.append(ds_val)

    if os.path.isdir(test_path):
        ds_test = torchvision.datasets.ImageFolder(root=test_path, transform=transform_train)
        train_datasets.append(ds_test)

if not train_datasets:
    print("Error: No datasets found.")
    sys.exit(1)

full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
full_val_dataset = torch.utils.data.ConcatDataset(val_datasets)

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
class_sample_counts = np.bincount(all_targets)
total_samples = len(all_targets)
class_weights = total_samples / (class_sample_counts + 1e-6)
print(f"Class Weights: {class_weights}")

samples_weights = class_weights[all_targets]
samples_weights = torch.from_numpy(samples_weights).double()

sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

# Optimized DataLoaders
train_loader = torch.utils.data.DataLoader(
    full_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
    drop_last=True, pin_memory=True, sampler=sampler, persistent_workers=True, prefetch_factor=4
)
val_loader = torch.utils.data.DataLoader(
    full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, 
    pin_memory=True, persistent_workers=True, prefetch_factor=4
)

# --- Target Validation Streams (Zero-Shot Tracking) ---
target_val_loaders = {}
target_datasets_list = ['KDEFFormated', 'CKplusIm', 'jaffeFormated']

print("Loading Target Validation Sets...")
for target in target_datasets_list:
    path = os.path.join(DATASET_ROOT, target)
    target_loader_path = None
    if os.path.isdir(os.path.join(path, 'test')): target_loader_path = os.path.join(path, 'test')
    elif os.path.isdir(os.path.join(path, 'eval')): target_loader_path = os.path.join(path, 'eval')
    elif os.path.isdir(os.path.join(path, 'train')): target_loader_path = os.path.join(path, 'train')
    
    if target_loader_path:
        ds_target = torchvision.datasets.ImageFolder(root=target_loader_path, transform=transform_val) 
        target_val_loaders[target] = torch.utils.data.DataLoader(
            ds_target, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4
        )
    else:
        print(f"Warning: Could not find any split for target {target}")


# ## 2. Model Initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CCT(
    img_size=IMG_SIZE,
    num_classes=len(CLASSES),
    positional_embedding='learnable',
    stochastic_depth=0.1, 
    kernel_size=3,
    stride=1, 
    padding=1
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# --- Optimizer: SAM wrapping AdamW ---
base_optimizer = torch.optim.AdamW
optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_val_acc = 0.0

def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch_index+1}")
    
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # --- SAM STEP 1 ---
        enable_running_stats(model)
        
        # AMP Forward 1
        with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        if torch.isnan(loss):
            print(f"Warning: NaN loss. Skipping.")
            optimizer.zero_grad()
            scaler.update() 
            continue
            
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUM_STEPS == 0:
            # Unscale 1
            try:
                scaler.unscale_(optimizer)
            except RuntimeError:
                pass
            
            # Check for Inf/NaN gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print("Warning: Infinite gradients in step 1. Skipping.")
                scaler.update()
                optimizer.zero_grad()
                continue

            optimizer.first_step(zero_grad=True)
            
            # --- SAM STEP 2 ---
            disable_running_stats(model) 
            
            # AMP Forward 2
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                outputs2 = model(inputs)
                loss2 = criterion(outputs2, labels)
            
            scaler.scale(loss2).backward()
            
            # Unscale 2
            try:
                scaler.unscale_(optimizer)
            except RuntimeError:
                pass
            
            grad_norm2 = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm2):
                print("Warning: Infinite gradients in step 2. Skipping.")
                scaler.update() 
                optimizer.zero_grad()
                continue
            
            optimizer.second_step(zero_grad=True)
            
            scaler.update()
            
            # Stats
            loss_val = loss.item()
            running_loss += loss_val * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total, 'lr': current_lr})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Train - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    return epoch_loss

def validate(epoch, loader, name="Global"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # TTA: Simple average of Original + Flipped
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                output_orig = model(inputs)
                output_flip = model(torch.flip(inputs, dims=[3]))
                outputs = (output_orig + output_flip) / 2.0
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    loss_avg = running_loss / total
    print(f"Validation ({name}) - Loss: {loss_avg:.4f}, Acc: {acc:.2f}%")
    return acc

# Main Loop
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run a short test')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint (model_last.pth)')
    args = parser.parse_args()

    if args.dry_run:
        print("Running Dry Run...")
        EPOCHS = 1
        # Truncate datasets for speed
        if len(full_train_dataset) > 0:
            # Create a tiny subset
            indices = torch.randperm(len(full_train_dataset))[:100]
            subset = torch.utils.data.Subset(full_train_dataset, indices)
            # Recreate loader
            train_loader = torch.utils.data.DataLoader(
                subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0 # No workers for speed
            )
            print(f"Dry Run: Modified train loader to {len(subset)} samples")

    save_dir = '/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/CCT-7withSAM/'
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, Best Acc: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, EPOCHS):
    
        train_loss = train_one_epoch(epoch)
    
        # Validate less frequently
        val_acc = 0.0
        if (epoch + 1) % VALIDATE_FREQ == 0 or (epoch + 1) == EPOCHS:
            # Validate Global
            val_acc = validate(epoch, val_loader, name="Global (Val)")
            
            # Validate Targets
            for name, loader in target_val_loaders.items():
                validate(epoch, loader, name=name)
                
            # Save Best (Based on Global)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc': best_val_acc
                }, os.path.join(save_dir, 'model_best.pth'))
                print(f"Saved Best Model: {best_val_acc:.2f}%")
                
            # Every 50 epochs save a checkpoint
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc': best_val_acc
                }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        else:
            print(f"Epoch {epoch+1}: Skipping Validation.")
            
        # Save Last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_val_acc
        }, os.path.join(save_dir, 'model_last.pth'))
