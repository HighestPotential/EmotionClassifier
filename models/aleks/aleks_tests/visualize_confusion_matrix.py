
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Constants
TEST_DATA_ROOT = "/home/d/dumanskyy/work/EmotionClassifier/test_datasets"
WEIGHTS_PATH = "/home/d/dumanskyy/work/EmotionClassifier/models/aleks/improved_v1/checkpoint.pth"
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMG_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 4

# Import Model
try:
    from aleks_resnet18_se import ResNet18SE
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from aleks_resnet18_se import ResNet18SE

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class RecursiveUniversalDataset(Dataset):
    def __init__(self, root_dirs, class_to_idx, transform=None):
        self.samples = []
        self.class_to_idx = class_to_idx
        self.transform = transform
        
        for root_path in root_dirs:
            # Walk through everything
            for dp, _, fns in os.walk(root_path):
                # Check if current folder name is a class
                folder_name = os.path.basename(dp)
                if folder_name in class_to_idx:
                    target = class_to_idx[folder_name]
                    for fn in fns:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}:
                            full_path = os.path.join(dp, fn)
                            self.samples.append((full_path, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot Count
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
    plt.title('Confusion Matrix (Count)', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Confusion Matrix saved to {output_path}")
    plt.close()

    # Plot Normalized
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
    plt.title('Confusion Matrix (Normalized)', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    norm_path = output_path.replace(".png", "_normalized.png")
    plt.savefig(norm_path, dpi=300)
    print(f"Normalized Confusion Matrix saved to {norm_path}")
    plt.close()

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Define Datasets to look for - CKplusIm and KDEF
    target_datasets = ["CKplusIm", "KDEFFormatedCroppedTest2"]
    root_dirs = []
    
    for name in target_datasets:
        path = os.path.join(TEST_DATA_ROOT, name)
        if os.path.isdir(path):
            root_dirs.append(path)
        else:
            print(f"Warning: {name} not found in {TEST_DATA_ROOT}")
            
    if not root_dirs:
        print("No datasets found!")
        return

    # Transforms
    eval_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = RecursiveUniversalDataset(root_dirs, CLASS_TO_IDX, transform=eval_tfm)
    print(f"Evaluating on {len(dataset)} images from {target_datasets}")

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # Model Setup
    model = ResNet18SE(num_classes=len(CLASSES)).to(device)
    
    # APPLY THE ARCHITECTURE FIX
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
         in_f = model.fc.in_features
         model.fc = nn.Sequential(
             nn.Dropout(p=0.3),
             nn.Linear(in_f, len(CLASSES))
         ).to(device)

    # Load Weights
    if os.path.isfile(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}...")
        try:
             ckpt = torch.load(WEIGHTS_PATH, map_location=device)
             if 'model_state_dict' in ckpt:
                 model.load_state_dict(ckpt['model_state_dict'])
             else:
                 model.load_state_dict(ckpt)
             print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    model.eval()
    
    y_true = []
    y_pred = []

    print("\nStarting Evaluation...")
    with torch.no_grad():
        for imgs, targets in tqdm(loader, unit="batch"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    output_file = "confusion_matrix_resnet18_all_test.png"
    plot_confusion_matrix(y_true, y_pred, CLASSES, output_file)

if __name__ == "__main__":
    main()
