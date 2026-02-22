import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from balanced_loss import Loss

try:
    import timm
    from timm.models import create_model
except ImportError:
    print("Error: 'timm' library is required. Please install it.")
    sys.exit(1)

# Constants
BATCH_SIZE = 16
ACCUM_STEPS = 4 
EPOCHS = 300
LR = 0.0005
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9 
IMG_SIZE = 64
NUM_WORKERS = 8
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_1_0_ready_to_use_datasets'
DATASETS = ['AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB', 'KDEFFormated',\
     'jaffeFormated', 'MMAFEDB', 'NONAMEFormated', 'ExpWFormated', 'EmoSet-118k']
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# --- Mixup & CutMix (Tuned for 64x64 faces) ---
cutmix = v2.CutMix(num_classes=len(CLASSES), alpha=0.3)
mixup = v2.MixUp(num_classes=len(CLASSES), alpha=0.2)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=[0.3, 0.7])

# --- Augmentation Settings ---
ROTATION_DEG = 15
TRANSLATE_FRAC = 0.05
COLOR_JITTER = 0.2
SHARPNESS_FACTOR = 2.0
GRAYSCALE_PROB = 0.3

# ## 1. Data Loading

transform_train = v2.Compose([
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
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    v2.RandomErasing(p=0.1, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0.0),
])

transform_eval = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize lists to hold dataset objects
train_datasets = []
val_datasets = []

# Helper to count samples
emotion_counts_train = {c: 0 for c in CLASSES}

print("Loading datasets...")
for ds_name in DATASETS:
    ds_path = os.path.join(DATASET_ROOT, ds_name)
    train_path = os.path.join(ds_path, 'train')
    eval_path = os.path.join(ds_path, 'eval')
    
    # Training Data
    if os.path.isdir(train_path):
        ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        train_datasets.append(ds)
        
        # Count for visualization and balanced loss
        for _, label_idx in ds.samples:
            class_name = ds.classes[label_idx]
            if class_name in emotion_counts_train:
                emotion_counts_train[class_name] += 1
    else:
        print(f"Warning: Train path not found for {ds_name}")

    # Validation Data
    if os.path.isdir(eval_path):
        ds_val = torchvision.datasets.ImageFolder(root=eval_path, transform=transform_eval)
        val_datasets.append(ds_val)
    else:
        print(f"Warning: Eval path not found for {ds_name}")

# Combine datasets
full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
full_val_dataset = torch.utils.data.ConcatDataset(val_datasets)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Total Training Samples: {len(full_train_dataset)}")
print(f"Total Validation Samples: {len(full_val_dataset)}")

samples_per_class = [emotion_counts_train[c] for c in CLASSES]
print("Samples per class:", samples_per_class)


# ## 2. Model Initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ConvNeXt V2 Atto
print("Initializing ConvNeXt V2 Atto...")
model = create_model('convnextv2_atto', pretrained=False, num_classes=len(CLASSES))

model.stem = nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1)

print("Model Stem modified for small image size.")
model = model.to(device)

# Loss Function
# Calculate Class Weights
total_samples = sum(samples_per_class)
class_weights = [total_samples / (len(CLASSES) * x) if x > 0 else 1.0 for x in samples_per_class]
print(f"Class Weights: {class_weights}")
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# --- AMP: Initialize GradScaler ---
scaler = torch.amp.GradScaler('cuda')

best_vloss = 1_000_000.

def train_one_epoch(epoch_index):
    model.train()
    
    # Track stats for the whole epoch
    epoch_loss = 0.
    epoch_correct = 0
    epoch_samples = 0
    
    # Track stats for the 5-batch interval logging
    running_loss = 0.
    running_correct = 0
    running_samples = 0
    
    # Tqdm for progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_index+1}")
    
    # Zero gradients at start of epoch
    optimizer.zero_grad()
    
    for i, data in pbar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        # Apply Mixup/CutMix
        inputs, labels = cutmix_or_mixup(inputs, labels)

        # --- AMP: Automatic Mixed Precision Context ---
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        # --- AMP: Scaled Backward Pass ---
        # Scale loss by 1/ACCUM_STEPS
        scaler.scale(loss / ACCUM_STEPS).backward()
        
        # Accumulate metrics
        loss_val = float(loss.item())
        epoch_loss += loss_val * batch_size
        running_loss += loss_val * batch_size
        
        # Calculate Accuracy (Start)
        # For soft targets, we need to convert back to hard labels for accuracy calculation
        _, preds = torch.max(outputs, 1)
        _, targets_hard = torch.max(labels, 1) # Argmax of the soft one-hot vectors
        
        correct_val = (preds == targets_hard).sum().item()
        epoch_correct += correct_val
        running_correct += correct_val
        epoch_samples += batch_size
        running_samples += batch_size

        # 3. Step: Only update weights after ACCUM_STEPS
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 

        # Log every 5 batches
        if (i + 1) % 5 == 0:
            avg_loss = running_loss / running_samples
            avg_acc = running_correct / running_samples
            
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}'})
            
            # Reset running counters
            running_loss = 0.
            running_correct = 0
            running_samples = 0

    # Calculate Epoch Averages
    avg_epoch_loss = epoch_loss / epoch_samples
    avg_epoch_acc = epoch_correct / epoch_samples
    
    print(f"Train - Loss: {avg_epoch_loss:.4f}, Acc: {avg_epoch_acc:.4f}")
    return avg_epoch_loss

def validate(epoch_index):
    model.eval()
    running_vloss = 0.0
    running_vaccuracy = 0.0
    total_val = 0
    
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            batch_size = vinputs.size(0)
            
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)

            running_vloss += vloss.item() * batch_size
            
            _, vpreds = torch.max(voutputs, 1)
            running_vaccuracy += (vpreds == vlabels).sum().item()
            total_val += batch_size

    avg_vloss = running_vloss / total_val
    avg_vacc = running_vaccuracy / total_val

    
    print(f"Validation - Loss: {avg_vloss:.4f}, Acc: {avg_vacc:.4f}")
    return avg_vloss

# Main Loop
for epoch in range(EPOCHS):
    print(f'EPOCH {epoch + 1}:')
    
    # Train
    avg_train_loss = train_one_epoch(epoch)
    
    # Validate
    avg_vloss = validate(epoch)
    
    # Checkpoint
    model_dir = '/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/ConvNeXt_V2_Atto'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'model_checkpoint_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_path)
    
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print(f"Saved model to {model_path} (New Best Validation Loss: {best_vloss:.4f})")
    else:
        print(f"Saved model to {model_path}")
