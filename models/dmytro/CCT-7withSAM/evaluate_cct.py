
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
from custom_cct import CCT
import argparse

# --- Configuration ---
IMG_SIZE = 64
BATCH_SIZE = 128
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform_val = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def validate(model, loader, name="Global"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # TTA: Simple average of Original + Flipped (Matches training script)
            output_orig = model(inputs)
            output_flip = model(torch.flip(inputs, dims=[3]))
            outputs = (output_orig + output_flip) / 2.0
            
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    loss_avg = running_loss / total
    print(f"Validation ({name}) - Loss: {loss_avg:.4f}, Acc: {acc:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth checkpoint')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # Model
    print("Loading CCT Model...")
    model = CCT(
        img_size=IMG_SIZE,
        num_classes=len(CLASSES),
        positional_embedding='learnable',
        stochastic_depth=0.1, 
        kernel_size=3,
        stride=1, 
        padding=1
    )
    model = model.to(DEVICE)

    # Load Weights
    if os.path.isfile(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=DEVICE)
        # Handle 'model_state_dict' key if present
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Weights file not found at {args.weights}")
        return

    # Datasets to Check
    # We look in ../datasets/ (Root of project/datasets)
    # Assuming script is in models/dmytro/CCT-7withSAM/
    # project_root = "/home/d/dumanskyy/work/EmotionClassifier" 
    # dataset_root = os.path.join(project_root, "datasets")
    dataset_root = "/home/d/dumanskyy/work/EmotionClassifier/test_datasets"

    targets = ['KDEFFormated', 'CKplusIm']
    
    print("\nStarting Evaluation explicitly on Zero-Shot targets...")
    
    for target in targets:
        path = os.path.join(dataset_root, target)
        
        # Try to find best split
        loader_path = None
        if os.path.isdir(os.path.join(path, 'test')): loader_path = os.path.join(path, 'test')
        elif os.path.isdir(os.path.join(path, 'eval')): loader_path = os.path.join(path, 'eval')
        elif os.path.isdir(os.path.join(path, 'train')): loader_path = os.path.join(path, 'train') # Fallback
        
        if loader_path:
            ds = torchvision.datasets.ImageFolder(root=loader_path, transform=transform_val)
            loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            print(f"\nEvaluating on {target} ({len(ds)} samples)...")
            validate(model, loader, name=target)
        else:
            print(f"Warning: Could not find valid split for {target} in {path}")

if __name__ == "__main__":
    main()
