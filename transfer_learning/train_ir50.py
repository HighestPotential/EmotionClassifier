import sys
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import numpy as np
from collections import Counter
from tqdm import tqdm
import timm
from timm.utils import ModelEmaV2, accuracy
from timm.scheduler import CosineLRScheduler

# Import IR50
try:
    from ir50 import Backbone, load_pretrained_weights
except ImportError:
    print("Error: Could not import ir50.py. Ensure it is in the same directory.")
    sys.exit(1)

# --- Configuration (MATCHING RESNET18 SCRIPT) ---
BATCH_SIZE = 32         
ACCUM_STEPS = 2          
EPOCHS = 25 
LR = 0.02                
WEIGHT_DECAY = 5e-4      # MATCHED: Standard SGD decay
MOMENTUM = 0.9
IMG_SIZE = 112           # REQUIRED: IR50 Native Resolution
NUM_WORKERS = 8
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# --- Paths ---
DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/GAN3112'
TEST_DATASET_ROOT = '/home/d/dumanskyy/work/EmotionClassifier/test_datasets'
PRETRAINED_WEIGHTS = '/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/pretrain/ir50.pth'
SAVE_DIR = '/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/checkpoints'
RESULTS_DIR = '/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/resultsSRGAN'

# --- Mixup Params (MATCHED) ---
MIXUP_ALPHA = 0.15       # MATCHED: ResNet18 uses 0.15
MIXUP_PROB = 0.3         # MATCHED: ResNet18 uses 0.3

# --- Augmentation Intensity (MATCHED) ---
ROTATION_DEG = 10        # MATCHED: ResNet18 uses 10 (via RandomAffine) or 15
TRANSLATE_FRAC = 0.06    # MATCHED: ResNet18 uses 0.06
COLOR_JITTER = 0.2       # MATCHED: ResNet18 uses 0.2
SHARPNESS_FACTOR = 2.0
GRAYSCALE_PROB = 0.3     # MATCHED: Standard is usually lower

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Loss Function (CBFL - Critical for imbalance) ---
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
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)          
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# --- Helper for Mixup ---
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Data Pipeline ---
def get_transforms():
    # MATCHED: ResNet18 structure
    # Resize -> Flip -> ColorJitter -> RandomAffine -> ToTensor -> Normalize -> RandomErasing
    
    # Note: V2 syntax used here for speed, but logic is matched.
    train_transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.RandomHorizontalFlip(p=0.5), # MATCHED: p=0.5
        v2.RandomApply([
            v2.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=0.15, hue=0.03)
        ], p=0.5), # MATCHED: p=0.5
        v2.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=3), # MATCHED
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0) # MATCHED: Validated from ResNet18 script
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return train_transform, val_transform

def count_samples(dataset):
    counts = [0] * len(CLASSES)
    # Recursively check dataset
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        ds_list = dataset.datasets
    else:
        ds_list = [dataset]
        
    for ds in ds_list:
        if hasattr(ds, 'targets'):
             # Standard ImageFolder
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
            if not os.path.isdir(ds_path): continue
            
            for sub in ['train', 'eval']:
                p = os.path.join(ds_path, sub)
                if os.path.isdir(p):
                    try:
                        train_datasets.append(torchvision.datasets.ImageFolder(p, transform=train_tfm))
                    except: pass

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
            if not os.path.isdir(ds_path): continue
            try:
                ds = torchvision.datasets.ImageFolder(ds_path, transform=val_tfm)
                val_datasets.append(ds)
            except: pass
            
    full_val_ds = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None

    # Loaders - SHUFFLE=TRUE (Matched ResNet18)
    train_loader = torch.utils.data.DataLoader(
        full_train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    
    val_loader = None
    if full_val_ds:
        val_loader = torch.utils.data.DataLoader(
            full_val_ds, batch_size=BATCH_SIZE, shuffle=False, 
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
    return train_loader, val_loader, samples_per_cls

def train(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    csv_path = os.path.join(RESULTS_DIR, 'ir50_cbfl_results.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "LR", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
    
    print("Initializing Model...")
    # Initialize BackBone (IR-50)
    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se', num_classes=len(CLASSES))
    
    # Load Pretrained Features
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading weights from {PRETRAINED_WEIGHTS}")
        load_pretrained_weights(model, torch.load(PRETRAINED_WEIGHTS, map_location='cpu'))
    
    model = model.to(device)
    model_ema = ModelEmaV2(model, decay=0.999)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    scheduler = CosineLRScheduler(
        optimizer, t_initial=EPOCHS, lr_min=1e-5, 
        warmup_t=5, warmup_lr_init=1e-4, warmup_prefix=True
    )

    train_loader, val_loader, samples_per_cls = load_data()
    print(f"Samples per class: {samples_per_cls}")
    
    train_criterion = ClassBalancedFocalLoss(samples_per_cls).to(device)
    val_criterion = nn.CrossEntropyLoss().to(device) 

    print(f"Starting training for {EPOCHS} epochs...")
    
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Manual Mixup (MATCHED)
            do_mixup = np.random.rand() < MIXUP_PROB
            if do_mixup:
                inputs, y_a, y_b, lam = mixup_batch(inputs, targets, alpha=MIXUP_ALPHA)
            
            # Forward
            outputs = model(inputs)
            
            if do_mixup:
                loss = mixup_loss(train_criterion, outputs, y_a, y_b, lam)
            else:
                loss = train_criterion(outputs, targets)
            
            # Gradient Accumulation
            loss = loss / ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                # Gradient Clipping (MATCHED: ResNet18 uses 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                model_ema.update(model)
            
            # Loss reporting (scale back up for log)
            run_loss += loss.item() * ACCUM_STEPS * inputs.size(0)
            
            # Simple acc check
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            
            pbar.set_postfix({'loss': run_loss/total, 'acc': 100.*correct/total})
            
        scheduler.step(epoch)
        
        # Validation
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
            
            # Save if best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model_ema.module.state_dict(), os.path.join(SAVE_DIR, 'ir50_best_cbfl.pth'))
                print(f"New Best Saved! ({best_acc:.2f}%)")
        
        # Log
        with open(csv_path, "a", newline="") as f:
             csv.writer(f).writerow([epoch+1, optimizer.param_groups[0]['lr'], run_loss/total, 100.*correct/total, val_loss_avg, val_acc])

if __name__ == "__main__":
    train(None)
