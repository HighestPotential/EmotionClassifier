
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

# Constants
TEST_DATA_ROOT = "/home/d/dumanskyy/work/EmotionClassifier/test_datasets"
WEIGHTS_PATH = "/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/EfficentNetV2_v2_sam/efficientnetv2_last.pth"
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMG_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 8
DROP_PATH_RATE = 0.3

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

def create_cifar_efficientnet():
    # Matching the definition in efficent_net_v3.py
    print("Creating EfficientNetV2-S...")
    model = timm.create_model(
        'tf_efficientnetv2_s', 
        pretrained=False, 
        num_classes=len(CLASSES), 
        drop_path_rate=DROP_PATH_RATE
    )
    
    # Modify Stem Stride to 1
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

def main():
    device = get_device()
    print(f"Using device: {device}")

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

    # Transforms (Mean/Std from efficent_net_v3.py)
    MEAN = [0.4681, 0.4447, 0.4560]
    STD = [0.2327, 0.2227, 0.2224]
    
    eval_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    dataset = RecursiveUniversalDataset(root_dirs, CLASS_TO_IDX, transform=eval_tfm)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # Initialize Model
    model = create_cifar_efficientnet().to(device)

    # Load Weights
    if os.path.isfile(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}...")
        try:
             ckpt = torch.load(WEIGHTS_PATH, map_location=device)
             
             if 'model_ema_state_dict' in ckpt:
                 print("Loading EMA weights...")
                 model.load_state_dict(ckpt['model_ema_state_dict'])
             elif 'model_state_dict' in ckpt:
                 print("Loading standard weights (EMA not found)...")
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
    dataset_correct = {name: 0 for name in dataset.dataset_stats}
    class_correct = {c: 0 for c in CLASSES}
    class_total = {c: 0 for c in CLASSES}

    print("\nStarting Evaluation with TTA (Horizontal Flip)...")
    with torch.no_grad():
        for imgs, targets, ds_names in tqdm(loader, unit="batch"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Original Inference
            logits = model(imgs)
            
            imgs_flip = torch.flip(imgs, [3])
            logits_flip = model(imgs_flip)
            
            # Average Logits
            avg_logits = (logits + logits_flip) / 2.0
            
            preds = avg_logits.argmax(dim=1)
            
            correct_mask = (preds == targets)
            total_correct += correct_mask.sum().item()
            total_samples += targets.size(0)
            
            for i in range(len(targets)):
                label = CLASSES[targets[i]]
                is_correct = correct_mask[i].item()
                
                class_total[label] += 1
                if is_correct:
                    class_correct[label] += 1
                    dataset_correct[ds_names[i]] += 1

    print("\n" + "="*40)
    print(f"Global Accuracy: {total_correct}/{total_samples} = {(total_correct/total_samples)*100:.2f}%")
    print("="*40)
    
    print("\nPer-Class Accuracy:")
    for cls in CLASSES:
        if class_total[cls] > 0:
            acc = 100. * class_correct[cls] / class_total[cls]
            print(f"  {cls:<10}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
        else:
            print(f"  {cls:<10}: N/A (0 samples)")
    
    print("\nPer-Dataset Breakdown:")
    
    grand_total_check = 0
    for name in sorted(dataset.dataset_stats.keys()):
        total = dataset.dataset_stats[name]["total"]
        corr = dataset_correct.get(name, 0)
        acc = (corr / total) * 100 if total > 0 else 0.0
        print(f"  {name:<25}: {corr}/{total} ({acc:.2f}%)")
        grand_total_check += total

if __name__ == "__main__":
    main()
