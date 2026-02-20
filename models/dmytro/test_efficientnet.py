import torch
import torch.nn as nn
import sys
import torchvision
import torchvision.transforms as transforms
import timm
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from timm.utils import accuracy

try:
    from batch_face import RetinaFace
    BATCH_FACE_AVAILABLE = True
except ImportError:
    BATCH_FACE_AVAILABLE = False


# --- Configuration (Copied from efficent_net_v2.py) ---
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
IMG_SIZE = 64
MEAN = [0.4681, 0.4447, 0.4560]
STD = [0.2327, 0.2227, 0.2224]
DROP_PATH_RATE = 0.2

# --- Model Factory (EfficientNetV2) ---
def create_cifar_efficientnet():
    print("Creating EfficientNetV2-S with Overfitting Fix...")
    model = timm.create_model(
        'tf_efficientnetv2_s', 
        pretrained=False, 
        num_classes=len(CLASSES), 
        drop_path_rate=DROP_PATH_RATE
    )
    
    # 1. Modify Stem Stride (The "64x64 Fix")
    old_stem = model.conv_stem
    model.conv_stem = nn.Conv2d(
        in_channels=old_stem.in_channels,
        out_channels=old_stem.out_channels,
        kernel_size=old_stem.kernel_size,
        stride=(1, 1), # CRITICAL FIX
        padding=old_stem.padding,
        bias=False
    )
    # Re-init stem
    nn.init.kaiming_normal_(model.conv_stem.weight, mode='fan_out', nonlinearity='relu')
    
    return model

# --- Model Factory (Vim) ---
def create_vim_model():
    print("Creating Vim-Tiny Model...")
    # Add Vim directory to sys.path to allow imports
    vim_path = '/home/d/dumanskyy/work/EmotionClassifier/models/dmytro/VIM/Vim/vim'
    if vim_path not in sys.path:
        sys.path.append(vim_path)
    
    try:
        from models_mamba import VisionMamba
    except ImportError as e:
        print(f"Error importing VisionMamba: {e}")
        print(f"Ensure that {vim_path} exists and dependencies (mamba_ssm, etc.) are installed.")
        sys.exit(1)

    model = VisionMamba(
        img_size=IMG_SIZE,
        patch_size=8,  # CHANGED from 16 to 8 based on checkpoint analysis
        stride=8,      # CHANGED to match patch_size
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        final_pool_type='mean', 
        if_abs_pos_embed=True, 
        if_rope=False, 
        if_rope_residual=False, 
        bimamba_type="v2", 
        if_cls_token=True, 
        if_divide_out=True, 
        use_middle_cls_token=True,
        num_classes=len(CLASSES)
    )
    return model

# --- Face Cropping Logic (Adapted from preprocessing/cropp_face.py) ---
class FaceCropper:
    def __init__(self, device='cuda'):
        self.device = device
        if BATCH_FACE_AVAILABLE:
            # gpu_id=0 if cuda, else -1. 
            gpu_id = 0 if device == 'cuda' else -1
            self.detector = RetinaFace(gpu_id=gpu_id)
        else:
            self.detector = None

# We need a custom Dataset or Transform to handle cropping BEFORE ToTensor/Normalize.
class CroppedDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, crop_faces=False, device='cuda'):
        try:
             super().__init__(root, transform=None) # We apply transform manually
        except Exception as e:
             # If root is not valid image folder structure, ImageFolder might fail init or return empty
             # We let it bubble or handle inside the try/except of evaluate_split
             raise e
             
        self.final_transform = transform
        self.crop_faces = crop_faces
        self.cropper = FaceCropper(device) if crop_faces and BATCH_FACE_AVAILABLE else None
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        # Load image using generic loader (PIL)
        sample = self.loader(path)
        
        if self.crop_faces and self.cropper and self.cropper.detector:
            try:
                # PIL -> Numpy RGB
                img_np = np.array(sample)
                # RetinaFace expects RGB? usually yes.
                
                faces = self.cropper.detector(img_np, threshold=0.9)
                if faces:
                    # Find largest face
                    # box: [x1, y1, x2, y2]
                    best_face = max(faces, key=lambda x: (x[0][2]-x[0][0]) * (x[0][3]-x[0][1]))
                    box, _, _ = best_face
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Padding (from cropp_face.py)
                    w = x2 - x1
                    h = y2 - y1
                    pad_w = int(w * 0.4)
                    pad_h = int(h * 0.2)
                    
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(img_np.shape[1], x2 + pad_w)
                    y2 = min(img_np.shape[0], y2 + pad_h)
                    
                    if x2 > x1 and y2 > y1:
                        # Crop
                        img_crop = img_np[y1:y2, x1:x2]
                        # Back to PIL
                        from PIL import Image
                        sample = Image.fromarray(img_crop)
            except Exception as e:
                # pass silently and use original
                pass
        
        # --- DEBUG: Save the first few processed images to verify crop ---
        if index < 10 and self.crop_faces:
             debug_dir = os.path.join(self.root, "param_debug_crops")
             os.makedirs(debug_dir, exist_ok=True)
             # Save raw sample (before transform, but after crop)
             save_path = os.path.join(debug_dir, f"crop_debug_{index}.jpg")
             try:
                 # If sample is PIL
                 sample.save(save_path)
             except:
                 pass
        # ---------------------------------------------------------------

        if self.final_transform is not None:
            sample = self.final_transform(sample)
            
        return sample, target

def evaluate_split(model, dataset_path, args, split_name="Dataset"):
    print(f"\n=== Evaluating {split_name} ===")
    print(f"Loading dataset from {dataset_path}")
    
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    try:
        # Custom Dataset
        dataset = CroppedDataset(
            root=dataset_path, 
            transform=transform_val,
            crop_faces=args.crop_faces,
            device=args.device
        )
    except Exception as e:
        print(f"Error creating dataset for {dataset_path}: {e}")
        return
        
    if len(dataset) == 0:
        print(f"No images found in {dataset_path}")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes.")

    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    class_correct = {c: 0 for c in dataset.classes}
    class_total = {c: 0 for c in dataset.classes}
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store for Confusion Matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Per-class accuracy
            for i in range(len(targets)):
                label = targets[i].item()
                pred = predicted[i].item()
                class_name = dataset.classes[label]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1

    acc = 100. * correct / total
    avg_loss = running_loss / total
    
    print(f"\n--- Results for {split_name} ---")
    print(f"Overall Accuracy: {acc:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
    print("\nPer-class Accuracy:")
    for class_name in dataset.classes:
        if class_total[class_name] > 0:
            class_acc = 100. * class_correct[class_name] / class_total[class_name]
            print(f"  {class_name}: {class_acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
        else:
            print(f"  {class_name}: N/A (0 samples)")
            
    print("\n--- Confusion Matrix ---")
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_targets, all_preds, labels=range(len(dataset.classes)))
        print(f"{'True/Pred':<12}", end="")
        for cls in dataset.classes:
            print(f"{cls[:6]:>8}", end="")
        print()
        
        for i, row in enumerate(cm):
            true_cls = dataset.classes[i]
            print(f"{true_cls:<12}", end="")
            for val in row:
                print(f"{val:8d}", end="")
            print()
    except ImportError:
        print("sklearn not installed, skipping confusion matrix.")

def main():
    parser = argparse.ArgumentParser(description='Test model on a dataset')
    parser.add_argument('weights', type=str, help='Path to the .pth weights file')
    parser.add_argument('--dataset', type=str, default='/home/d/dumanskyy/work/EmotionClassifier/new_preprocessed_dataset/WSEFEPFormated', 
                        help='Path to the dataset directory')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'vim'],
                        help='Model architecture to use')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--crop-faces', action='store_true', help='Enable on-the-fly face cropping (requires batch_face)')
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        return
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory not found at {args.dataset}")
        return

    if args.crop_faces and not BATCH_FACE_AVAILABLE:
        print("Warning: --crop-faces requested but 'batch_face' library not found. Running WITHOUT cropping.")
        args.crop_faces = False

    print(f"Using device: {args.device}")
    
    # Model Setup
    try:
        if args.model == 'efficientnet':
            model = create_cifar_efficientnet()
        elif args.model == 'vim':
            model = create_vim_model()
        else:
            print(f"Unknown model: {args.model}")
            return
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    model = model.to(args.device)
    model.eval()
    
    # Load Weights matching check_val_loss.py logic
    print(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=args.device)
    
    # Logic: Prefer EMA, then Model
    if 'model_ema_state_dict' in checkpoint:
        print("Found EMA state dict, using it.")
        state_dict = checkpoint['model_ema_state_dict']
    elif 'model_state_dict' in checkpoint:
        print("Found model state dict.")
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        print("Found model state dict (model key).")
        state_dict = checkpoint['model']
    else:
        # Assume the whole file is the state dict
        state_dict = checkpoint
        
    # Strip prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load with strict=False to be safe (EfficientNet V2 S shouldn't have missing keys if architecture matches)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing Keys: {len(missing)}")
        if len(missing) < 10: print(f"Missing: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected Keys: {len(unexpected)}")
        
    print("Weights loaded successfully.")
    
    # --- Auto-Discovery of Valid Datasets ---
    # We want to find folders that contain the emotion classes (e.g. 'anger', 'happiness')
    # and evaluate them.
    
    dirs_to_evaluate = []
    
    # Helper to check if a folder is a valid dataset (contains emotion classes)
    def is_valid_dataset(path):
        if not os.path.isdir(path): return False
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # Check if at least one class matches our target CLASSES
        return any(c in subdirs for c in CLASSES)

    # 1. Check Root
    if is_valid_dataset(args.dataset):
        dirs_to_evaluate.append((os.path.basename(args.dataset), args.dataset))
        
    # 2. Check 1 Level Deep (e.g. dataset/train)
    try:
        level1 = [os.path.join(args.dataset, d) for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))]
        for d1 in level1:
            if is_valid_dataset(d1):
                name = os.path.relpath(d1, args.dataset)
                dirs_to_evaluate.append((name, d1))
            
            # 3. Check 2 Levels Deep (e.g. test_datasets/KDEF/eval)
            try:
                level2 = [os.path.join(d1, d) for d in os.listdir(d1) if os.path.isdir(os.path.join(d1, d))]
                for d2 in level2:
                    if is_valid_dataset(d2):
                        name = os.path.relpath(d2, args.dataset)
                        dirs_to_evaluate.append((name, d2))
            except OSError:
                pass
    except OSError:
        pass
        
    unique_dirs = sorted(list(set(dirs_to_evaluate)), key=lambda x: x[0])
    
    if not unique_dirs:
        print(f"No valid emotion datasets found in {args.dataset}. Looking for folders containing: {CLASSES}")
        return

    print(f"Found {len(unique_dirs)} datasets/splits to evaluate:")
    for name, _ in unique_dirs:
        print(f" - {name}")
        
    for name, path in unique_dirs:
        evaluate_split(model, path, args, split_name=name)

if __name__ == '__main__':
    main()
