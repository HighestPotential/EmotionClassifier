
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# --- Import Vision Mamba ---
# Path hack to find the Vim module
current_dir = os.path.dirname(os.path.abspath(__file__))
vim_root = os.path.join(current_dir, "VIM", "Vim")
vim_package = os.path.join(vim_root, "vim")

sys.path.append(vim_root)
sys.path.append(vim_package)

try:
    from vim.models_mamba import VisionMamba
except ImportError:
    try:
        from models_mamba import VisionMamba
    except ImportError:
        print("Error: Could not import VisionMamba. Check paths.")
        sys.exit(1)

# --- Constants ---
TEST_DATA_ROOT = "/home/d/dumanskyy/work/EmotionClassifier/test_datasets"
WEIGHTS_PATH = "/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/VIM/Vim/results/vim_tiny_last.pth"
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMG_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 8

# Mean/Std from training (approximate from EfficientNet run on same data)
MEAN = [0.5176, 0.4511, 0.4225]
STD = [0.2328, 0.2180, 0.2125]

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
        self.dataset_stats = {} 

        for root_path in root_dirs:
            dataset_name = os.path.basename(root_path)
            if dataset_name not in self.dataset_stats:
                self.dataset_stats[dataset_name] = {"total": 0, "correct": 0}
            
            for dp, _, fns in os.walk(root_path):
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

def create_vim_model():
    print("Creating Vision Mamba (Vim-Tiny)...")
    model = VisionMamba(
        img_size=IMG_SIZE,
        patch_size=8,
        stride=8,
        embed_dim=192,
        depth=24,
        channels=3,
        num_classes=len(CLASSES),
        drop_path_rate=0.0, # No drop path for eval
        use_middle_cls_token=True,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type='mean',
        if_abs_pos_embed=True,
        bimamba_type="v2",
    )
    return model

def safe_load_weights(model, path, device):
    if not os.path.exists(path):
        print(f"Error: Weights file not found at {path}")
        return False
        
    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Handle EMA weights if present
    if 'model_ema' in checkpoint:
        print("Loading EMA weights...")
        state_dict = checkpoint['model_ema']

    # Remove 'module.' prefix if DDP was used
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Resize positional embeddings if needed (common for Vim/ViT)
    if 'pos_embed' in new_state_dict and model.pos_embed.shape != new_state_dict['pos_embed'].shape:
        print(f"Resizing pos_embed: {new_state_dict['pos_embed'].shape} -> {model.pos_embed.shape}")
        # Simplistic resize - usually requires interpolation
        # For now, just warn and strict=False
        pass

    try:
        model.load_state_dict(new_state_dict, strict=False) # strict=False to handle potential minor mismatches
        print("Weights loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Datasets
    target_datasets = ["CKplusIm", "KDEFFormatedCroppedTest2"] # Add others if needed
    root_dirs = []
    
    for name in target_datasets:
        path = os.path.join(TEST_DATA_ROOT, name)
        if os.path.isdir(path):
            root_dirs.append(path)
        else:
            print(f"Warning: Dataset {name} not found at {path}")

    if not root_dirs:
        print(f"No valid datasets found in {TEST_DATA_ROOT}")
        return

    # Transforms (match training)
    # TTA: Standard + Horizontal Flip
    base_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # Apply base_transform in the dataset so DataLoader gets Tensors
    test_dataset = RecursiveUniversalDataset(
        root_dirs=root_dirs, 
        class_to_idx=CLASS_TO_IDX, 
        transform=base_transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Model
    model = create_vim_model()
    model = model.to(device)
    
    if not safe_load_weights(model, WEIGHTS_PATH, device):
        print("Using random initialization (Warning!)")

    model.eval()
    
    # Metrics
    correct = 0
    total = 0
    class_correct = {c: 0 for c in CLASSES}
    class_total = {c: 0 for c in CLASSES}
    
    dataset_correct = {name: 0 for name in test_dataset.dataset_stats}
    dataset_total = {name: 0 for name in test_dataset.dataset_stats}

    print("\nStarting evaluation (with TTA)...")
    
    with torch.no_grad():
        for i, (imgs, targets, ds_names) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # TTA: Forward pass original
            output_orig = model(imgs)
            
            # TTA: Forward pass flipped (flip width dim, usually last for NCHW)
            imgs_flip = torch.flip(imgs, dims=[-1]) 
            output_flip = model(imgs_flip)
            
            # Average predictions
            outputs = (output_orig + output_flip) / 2.0
            
            preds = outputs.argmax(dim=1)
            
            # Update metrics
            c = (preds == targets).squeeze()
            for j in range(len(targets)):
                label = CLASSES[targets[j]]
                is_correct = c[j].item()
                
                class_correct[label] += is_correct
                class_total[label] += 1
                
                ds_name = ds_names[j]
                dataset_correct[ds_name] += is_correct
                dataset_total[ds_name] += 1
                
                if is_correct:
                    correct += 1
                total += 1

    print("\n" + "="*40)
    print(f"Overall Accuracy: {100. * correct / total:.2f}% ({correct}/{total})")
    print("="*40)
    
    print("\nPer-Class Accuracy:")
    for cls in CLASSES:
        if class_total[cls] > 0:
            acc = 100. * class_correct[cls] / class_total[cls]
            print(f"  {cls:<10}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
        else:
            print(f"  {cls:<10}: N/A (0 samples)")

    print("\nPer-Dataset Accuracy:")
    for name in dataset_correct:
        if dataset_total[name] > 0:
            acc = 100. * dataset_correct[name] / dataset_total[name]
            print(f"  {name:<25}: {acc:.2f}% ({dataset_correct[name]}/{dataset_total[name]})")
        else:
            print(f"  {name:<25}: N/A")
            
    print("="*40)

if __name__ == "__main__":
    main()
