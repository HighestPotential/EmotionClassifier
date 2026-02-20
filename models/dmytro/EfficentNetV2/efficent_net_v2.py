#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
# Removed TensorBoard to prevent TensorFlow conflicts/crashesw
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from balanced_loss import Loss

# Ensure src is importable using the symlink
if os.path.islink('src') or os.path.exists('src'):
    # It should work directly if symlink exists in the boolean path
    pass
else:
    # Fallback to appending path if symlink is missing (though we created it)
    project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    compact_transformers_path = os.path.join(project_root, 'Compact-Transformers')
    if compact_transformers_path not in sys.path:
        sys.path.append(compact_transformers_path)


# Constants
BATCH_SIZE = 16 # Physical Batch Size (Small, for low VRAM)
ACCUM_STEPS = 4 # Aggregate gradients over 16 steps (Virtual Batch Size = 4 * 16 = 64)
EPOCHS = 300
LR = 0.01
MOMENTUM = 0.9
IMG_SIZE = 64
NUM_WORKERS = 8
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/ready_to_use_datasets'
DATASETS = ['AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB']
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Set seeds
torch.manual_seed(42)
np.random.seed(42)



transform = transforms.Compose([
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
        ds = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
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
        ds_val = torchvision.datasets.ImageFolder(root=eval_path, transform=transform)
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
plt.figure(figsize=(10, 6))
plt.bar(emotion_counts_train.keys(), emotion_counts_train.values(), color='skyblue')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.title('Distribution of Emotions in Training Set')
plt.show()

# The 'CLASSES' list above matches standard folder names, let's verify map
samples_per_class = [emotion_counts_train[c] for c in CLASSES]
print("Samples per class:", samples_per_class)


# ## 2. Model Initialization

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torchvision.models as models

def get_scratch_mobilenet_small(num_classes=6):
    # weights=None prevents loading ImageNet weights.
    model = models.mobilenet_v3_small(weights=None)
    
    # --- Structural Optimization for 64x64 ---
    # Access the first convolution layer
    first_conv = model.features[0][0]
    
    #  Re-initialize the first layer with stride=1.
    model.features[0][0] = nn.Conv2d(
        in_channels=first_conv.in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=1,
        padding=first_conv.padding,
        bias=False
    )
    
    in_features = model.classifier[3].in_features
    # Final projection to exactly 6 classes.
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

model = get_scratch_mobilenet_small(num_classes=len(CLASSES))
model = model.to(device)





criterion = Loss(
    loss_type="cross_entropy",
    samples_per_class=samples_per_class,
    class_balanced=True
)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# --- AMP: Initialize GradScaler ---
# GradScaler is essential for mixed precision training.
# It scales the loss to prevent underflow (vanishing gradients) when using float16.
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
    
    for i, data in pbar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        # Gradient Accumulation: Only zero_grad at the start of an accumulation cycle
        if i % ACCUM_STEPS == 0:
            optimizer.zero_grad()
        
        # --- AMP: Automatic Mixed Precision Context ---
        # "autocast" automatically casts operations to float16 where safe (like matrix muls)
        # and keeps others in float32 (like softmax/loss) for stability.
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        # --- AMP: Scaled Backward Pass ---
        # 1. Scale the loss: Multiplies loss by a large factor to keep gradients in representable range of float16.
        #    Gradient Accumulation: We must scale the loss by 1/ACCUM_STEPS so the sum of gradients is average.
        scaler.scale(loss / ACCUM_STEPS).backward()
        
        # 3. Step: Only update weights after ACCUM_STEPS
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # Optional: zero here or at start (safer at start usually, but here is standard)

        # Accumulate Loss and Accuracy 
        # IMPORTANT: Explicitly cast to python float() to detach from the computational graph.
        # If we just do `loss.item()`, sometimes PyTorch might hold onto the graph history, causing VRAM leaks.
        loss_val = float(loss.item())
        
        # Epoch aggregates (weight by batch_size)
        epoch_loss += loss_val * batch_size
        running_loss += loss_val * batch_size
        
        _, preds = torch.max(outputs, 1)
        correct_val = (preds == labels).sum().item()
        epoch_correct += correct_val
        running_correct += correct_val
        
        epoch_samples += batch_size
        running_samples += batch_size

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
            
            # Weighted aggregation
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
    
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/EfficentNetV2/best_model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path} (New Best Validation Loss: {best_vloss:.4f})")

    else:
        model_path = f'/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/EfficentNetV2/model_checkpoint_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
