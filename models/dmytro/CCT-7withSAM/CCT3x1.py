import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from balanced_loss import Loss

if os.path.islink('src') or os.path.exists('src'):
    pass
else:
    # Fallback to appending path if symlink is missing (though we created it)
    project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    compact_transformers_path = os.path.join(project_root, 'Compact-Transformers')
    if compact_transformers_path not in sys.path:
        sys.path.append(compact_transformers_path)

try:
    from custom_cct import CCT
except ImportError:
    # Explicit python path hack if running from root without package install
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from custom_cct import CCT

# Constants
BATCH_SIZE = 16
ACCUM_STEPS = 4 # Aggregate gradients over 16 steps (Virtual Batch Size = 4 * 16 = 64)
EPOCHS = 300
LR = 0.0005
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9 # Not used for AdamW but kept for reference if needed
IMG_SIZE = 64
NUM_WORKERS = 8
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/latest_1_0_ready_to_use_datasets'
DATASETS = ['AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB', 'KDEFFormated',\
     'jaffeFormated', 'MMAFEDB', 'NONAMEFormated', 'ExpWFormated', 'EmoSet-118k']
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Set seeds
torch.manual_seed(42)
np.random.seed(42)



# ## 1. Data Loading and Visualization
# 
# We will load datasets from the `ready_to_use_datasets` folder. We use `train` for training and `eval` for validation. `test` is reserved.

# CHANGED: Added robust augmentations for CCT training
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5), # Essential
    transforms.RandomRotation(degrees=15),  # Head tilt simulation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Robustness to lighting
    transforms.ToTensor(),
    # Normalize to [-1, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Validation stays clean
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Normalize to [-1, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        ds_val = torchvision.datasets.ImageFolder(root=eval_path, transform=transform_val)
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

# --- Visualization ---
# plt.figure(figsize=(10, 6))
# plt.bar(emotion_counts_train.keys(), emotion_counts_train.values(), color='skyblue')
# plt.xlabel('Emotion')
# plt.ylabel('Number of Samples')
# plt.title('Distribution of Emotions in Training Set')
# plt.show()

# Prepare samples_per_class list for Balanced Loss
samples_per_class = [emotion_counts_train[c] for c in CLASSES]
print("Samples per class:", samples_per_class)


# ## 2. Model Initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Using Custom CCT Implementation ---
model = CCT(
    img_size=IMG_SIZE,
    num_classes=len(CLASSES),
    positional_embedding='learnable',
    stochastic_depth=0.1,    # Technique B: Stochastic Depth 0.1
    kernel_size=3,
    stride=1,                # Technique C: Stride 1
    padding=1
)
model = model.to(device)

criterion = Loss(
    loss_type="cross_entropy",
    samples_per_class=samples_per_class,
    class_balanced=True
)

# Using AdamW as recommended for CCT/Transformers
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
        
        _, preds = torch.max(outputs, 1)
        correct_val = (preds == labels).sum().item()
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
    model_path = f'/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/CCT-7withSAM/model_checkpoint_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), model_path)
    
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print(f"Saved model to {model_path} (New Best Validation Loss: {best_vloss:.4f})")
    else:
        print(f"Saved model to {model_path}")
