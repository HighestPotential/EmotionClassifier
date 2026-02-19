
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, BatchNorm1d, Dropout, AdaptiveAvgPool2d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import IR50 backbone
try:
    from ir50 import Backbone, Flatten
except ImportError:
    print("Error: Could not import ir50.py. Ensure it is in the same directory.")
    sys.exit(1)

# Constants
TEST_DATA_ROOT = "/home/d/dumanskyy/work/EmotionClassifier/test_datasets"
WEIGHTS_PATH = "/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/results_light_head/ir50_light_head_best.pth"
CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMG_SIZE = 112  # IR50 native resolution
BATCH_SIZE = 128
NUM_WORKERS = 8

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


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


# ── Spatial Attention + Light Head (must match training exactly) ──────────────

class SpatialAttention(nn.Module):
    """Channel-wise spatial attention: learns which 7×7 regions matter."""
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.attn(x)


def replace_head_with_light(model, num_classes, drop_ratio=0.2):
    """Replace the heavy FC head with attention + GAP head (~795K params)."""
    model.output_layer = nn.Sequential(
        SpatialAttention(512),
        BatchNorm2d(512),
        AdaptiveAvgPool2d(1),
        Flatten(),
        Dropout(drop_ratio),
        nn.Linear(512, 512),
        BatchNorm1d(512),
        nn.ReLU(inplace=True),
        Dropout(0.15),
        nn.Linear(512, 512),
        BatchNorm1d(512),
        nn.ReLU(inplace=True),
        Dropout(0.1),
        nn.Linear(512, num_classes),
    )


def create_ir50_model():
    print("Creating IR50 with light head...")
    model = Backbone(50, 0.4, 'ir_se', num_classes=len(CLASSES))
    replace_head_with_light(model, len(CLASSES), drop_ratio=0.2)
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
    model = create_ir50_model().to(device)

    # Load Weights
    if os.path.isfile(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}...")
        try:
            ckpt = torch.load(WEIGHTS_PATH, map_location=device)

            # Try EMA weights first (usually better for validation)
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

    print("\nStarting Evaluation with TTA (Horizontal Flip)...")
    with torch.no_grad():
        for imgs, targets, ds_names in tqdm(loader, unit="batch"):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Original Inference
            logits = model(imgs)

            # TTA: Horizontal Flip
            imgs_flip = torch.flip(imgs, [3])
            logits_flip = model(imgs_flip)

            # Average Logits
            avg_logits = (logits + logits_flip) / 2.0

            preds = avg_logits.argmax(dim=1)

            correct_mask = (preds == targets)
            total_correct += correct_mask.sum().item()
            total_samples += targets.size(0)

            for i, correct in enumerate(correct_mask):
                if correct:
                    dataset_correct[ds_names[i]] += 1

    print("\n" + "=" * 40)
    print(f"Global Accuracy: {total_correct}/{total_samples} = {(total_correct/total_samples)*100:.2f}%")
    print("=" * 40)
    print("Per Dataset Breakdown:")

    for name in sorted(dataset.dataset_stats.keys()):
        total = dataset.dataset_stats[name]["total"]
        corr = dataset_correct.get(name, 0)
        acc = (corr / total) * 100 if total > 0 else 0.0
        print(f"{name:<25}: {corr}/{total} ({acc:.2f}%)")


if __name__ == "__main__":
    main()
