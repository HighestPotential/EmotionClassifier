
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Constants
TEST_DATA_ROOT = "/home/d/dumanskyy/work/EmotionClassifier/test_datasets"
WEIGHTS_PATH = "/home/d/dumanskyy/work/EmotionClassifier/models/aleks/original_adjusted/best_resnet18_se_SGD_cbfl_no_ckplus.pth.zip"
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
        self.dataset_stats = {}  # "DatasetName": {"total": 0, "correct": 0}

        # We will track which dataset each sample belongs to for stats
        # Assuming root_dirs is a list of full paths like [.../CKplusIm, .../KDEF...]
        
        for root_path in root_dirs:
            dataset_name = os.path.basename(root_path)
            if dataset_name not in self.dataset_stats:
                self.dataset_stats[dataset_name] = {"total": 0, "correct": 0} # correct will be filled during eval
            
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
                            self.samples.append((full_path, target, dataset_name))
                            self.dataset_stats[dataset_name]["total"] += 1

        print(f"Found {len(self.samples)} images total.")
        for name, stats in self.dataset_stats.items():
            print(f"  - {name}: {stats['total']} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target, dataset_name = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target, dataset_name

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Define Datasets to look for
    target_datasets = ["CKplusIm", "KDEFFormatedCroppedTest2"]
    root_dirs = []
    
    if not os.path.isdir(TEST_DATA_ROOT):
        print(f"Error: {TEST_DATA_ROOT} does not exist.")
        return

    # Find the target datasets in TEST_DATA_ROOT
    # Also support if the user gave absolute paths or if they are in subfolders?
    # Based on previous ls, they are direct children.
    
    # Actually, we can just walk TEST_DATA_ROOT and if we hit a folder matching target_datasets, we use it?
    # Or just construct path.
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

    # Dataset & Loader
    # We need a custom collate or just handle the tuple (img, target, ds_name) manually
    # Default collate handles strings in list? Yes.
    dataset = RecursiveUniversalDataset(root_dirs, CLASS_TO_IDX, transform=eval_tfm)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # Model Setup
    model = ResNet18SE(num_classes=len(CLASSES)).to(device)
    
    # APPLY THE ARCHITECTURE FIX from training script
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
             # Handle if it's a full checkpoint or just state dict
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
    
    total_correct = 0
    total_samples = 0
    
    # Store per-dataset correct counts
    dataset_correct = {name: 0 for name in dataset.dataset_stats}

    print("\nStarting Evaluation with TTA (Horizontal Flip)...")
    with torch.no_grad():
        for imgs, targets, ds_names in tqdm(loader, unit="batch"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Original
            logits = model(imgs)
            
            # Flip TTA
            imgs_flip = torch.flip(imgs, [3])
            logits_flip = model(imgs_flip)
            
            # Average
            avg_logits = (logits + logits_flip) / 2.0
            
            preds = avg_logits.argmax(dim=1)
            
            # Vectorized comparison
            correct_mask = (preds == targets)
            total_correct += correct_mask.sum().item()
            total_samples += targets.size(0)
            
            # Update per-dataset stats
            # This is a bit slow if batch size is huge and many datasets, but we only have 2.
            # Convert ds_names tuple to list for easier processing or just iterate
            # ds_names is a tuple of strings of length batch_size
            for i, correct in enumerate(correct_mask):
                if correct:
                    dataset_correct[ds_names[i]] += 1

    print("\n" + "="*40)
    print(f"Global Accuracy: {total_correct}/{total_samples} = {(total_correct/total_samples)*100:.2f}%")
    print("="*40)
    print("Per Dataset Breakdown:")
    
    grand_total_check = 0
    for name in sorted(dataset.dataset_stats.keys()):
        total = dataset.dataset_stats[name]["total"]
        corr = dataset_correct.get(name, 0)
        acc = (corr / total) * 100 if total > 0 else 0.0
        print(f"{name:<25}: {corr}/{total} ({acc:.2f}%)")
        grand_total_check += total

    assert grand_total_check == total_samples, "Sanity check failed: Sample count mismatch"

if __name__ == "__main__":
    main()
