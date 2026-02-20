
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from timm.utils import ModelEmaV2

# Config matching efficent_net_v2.py
IMG_SIZE = 64
MEAN = [0.4681, 0.4447, 0.4560]
STD = [0.2327, 0.2227, 0.2224]
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
WEIGHTS = '/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/EfficentNetV2/efficientnetv2_last.pth'
VAL_DIR = '/home/d/dumanskyy/work/test_datasets/jaffeFormated/eval'

def create_model():
    print("Creating EfficientNetV2-S...")
    model = timm.create_model(
        'tf_efficientnetv2_s', 
        pretrained=False, 
        num_classes=len(CLASSES), 
        drop_path_rate=0.2
    )
    # Stem Fix
    old_stem = model.conv_stem
    model.conv_stem = nn.Conv2d(
        in_channels=old_stem.in_channels,
        out_channels=old_stem.out_channels,
        kernel_size=old_stem.kernel_size,
        stride=(1, 1),
        padding=old_stem.padding,
        bias=False
    )
    return model

def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Pipeline
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    print(f"Loading validation set from {VAL_DIR}")
    dataset = torchvision.datasets.ImageFolder(root=VAL_DIR, transform=transform_val)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    print(f"Found {len(dataset)} images. Classes: {dataset.classes}")
    
    # 2. Model
    model = create_model().to(device)
    model.eval()
    
    # 3. Load Weights
    print(f"Loading weights from {WEIGHTS}")
    checkpoint = torch.load(WEIGHTS, map_location=device)
    
    # Try EMA first as per training script
    if 'model_ema_state_dict' in checkpoint:
        print("Loading EMA weights...")
        state_dict = checkpoint['model_ema_state_dict']
    else:
        print("Loading standard weights...")
        state_dict = checkpoint['model_state_dict']
        
    # Strip prefix
    new_state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing: 
        print(f"Missing: {len(missing)}")
        if len(missing) > 10: print(f"Sample: {missing[:5]}")
    if unexpected: print(f"Unexpected: {len(unexpected)}")
    
    # 4. Loop
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    avg_loss = running_loss / total
    
    print(f"\nResults for AffectNet/eval:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    validate()
