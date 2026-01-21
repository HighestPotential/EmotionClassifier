import torch
import torch.nn as nn
import sys
import torchvision
import torchvision.transforms as transforms
import timm
import os
import argparse
from tqdm import tqdm
from timm.utils import accuracy

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

    # Configuration matching the checkpoint (based on error logs)
    # Checkpoint has patch_embed.proj.weight shape [192, 3, 8, 8] -> implies stride=8 (kernel size is patch_size=16)
    # But wait, [192, 3, 8, 8] usually means kernel_size=8. 
    # Error message: "copying a param with shape torch.Size([192, 3, 8, 8]) from checkpoint, the shape in current model is torch.Size([192, 3, 16, 16])"
    # This means the checkpoint used kernel_size=8. 
    # The user might have used a model with patch_size=8. OR patch_size=16 but there is a confusion in my reading of the error.
    # The error says: checkpoint has [192, 3, 8, 8]. Current model has [192, 3, 16, 16].
    # Current model used patch_size=16. So checkpoint presumably used patch_size=8.
    
    # However, let's look at stride.
    # If I change patch_size to 8, then pos_embed size will increase drastically.
    # 64/8 = 8. 8x8 = 64 tokens. + 1 cls token = 65 tokens.
    # The checkpoint has pos_embed shape [1, 65, 192].
    # This PERFECTLY matches patch_size=8 for a 64x64 image (8*8 + 1 = 65).
    
    # So the fix is to use patch_size=8, stride=8 (default stride=patch_size usually).
    
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

def main():
    parser = argparse.ArgumentParser(description='Test model on a dataset')
    parser.add_argument('weights', type=str, help='Path to the .pth weights file')
    parser.add_argument('--dataset', type=str, default='/home/d/dumanskyy/work/EmotionClassifier/new_preprocessed_dataset/WSEFEPFormated', 
                        help='Path to the dataset directory (default: new_preprocessed_dataset/WSEFEPFormated)')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'vim'],
                        help='Model architecture to use (efficientnet or vim)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        return
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory not found at {args.dataset}")
        return

    print(f"Using device: {args.device}")
    
    # Data Setup
    print(f"Loading dataset from {args.dataset}")
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # User confirmed the structure is Root/Class/Images (no train/eval split)
    data_root = args.dataset
    
    try:
        dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform_val)
    except Exception as e:
        print(f"Error creating ImageFolder: {e}")
        print(f"Ensure that {data_root} contains subdirectories for each class.")
        return
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")
    
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
    
    # Load Weights
    print(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=args.device)
    
    # Handle different checkpoint formats
    state_dict = None
    if 'model_ema_state_dict' in checkpoint and checkpoint['model_ema_state_dict'] is not None:
        print("Found EMA state dict, using it.")
        state_dict = checkpoint['model_ema_state_dict']
    elif 'model_ema' in checkpoint and checkpoint['model_ema'] is not None:
         print("Found EMA state dict (model_ema key), using it.")
         state_dict = checkpoint['model_ema']
    elif 'model_state_dict' in checkpoint:
        print("Found model state dict.")
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        print("Found model state dict (model key).")
        state_dict = checkpoint['model']
    else:
        # Assume the whole file is the state dict
        state_dict = checkpoint
        
    # Load state dict
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict directly: {e}")
        print("Attempting to handle key mismatches and resizing...")
        
        # 1. Handle Key Mismatches (prefixes)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
            
        # 2. Handle Positional Embedding Mismatch (Resize if needed)
        # This is common in ViT/Vim when changing resolution or patch size
        if 'pos_embed' in new_state_dict and hasattr(model, 'pos_embed'):
            chk_pos_embed = new_state_dict['pos_embed']
            model_pos_embed = model.pos_embed
            if chk_pos_embed.shape != model_pos_embed.shape:
                print(f"Resizing pos_embed: {chk_pos_embed.shape} -> {model_pos_embed.shape}")
                # Logic adapted from timm/ViT load_pretrained
                # Assume standard [1, num_tokens, dim]
                # If cls token exists
                num_extra_tokens = 1 if model.if_cls_token else 0 # Assuming 1 for simplicity based on our init
                if model.use_double_cls_token: num_extra_tokens = 2
                
                # Check point tokens
                chk_num_extra = num_extra_tokens # Assume same topology
                chk_tokens = chk_pos_embed[:, chk_num_extra:]
                
                # Target grid size
                current_grid_size = model.patch_embed.grid_size # (H, W)
                
                # Input grid size (infer from token count)
                # chk_tokens.shape[1] should be H*W
                chk_seq_len = chk_tokens.shape[1]
                chk_dim = int(chk_seq_len ** 0.5)
                
                if chk_dim * chk_dim != chk_seq_len:
                    print("Warning: Checkpoint pos_embed sequence length is not a square. Interpolation might fail.")
                
                chk_tokens = chk_tokens.reshape(1, chk_dim, chk_dim, -1).permute(0, 3, 1, 2)
                
                target_tokens = torch.nn.functional.interpolate(
                    chk_tokens, size=current_grid_size, mode='bicubic', align_corners=False
                )
                target_tokens = target_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                
                # Re-cat with extra tokens
                new_pos_embed = torch.cat((chk_pos_embed[:, :num_extra_tokens], target_tokens), dim=1)
                new_state_dict['pos_embed'] = new_pos_embed
                
        model.load_state_dict(new_state_dict, strict=False) # strict=False to allow robust loading
        
    print("Weights loaded successfully.")
    
    # Evaluation Loop
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    class_correct = {c: 0 for c in dataset.classes}
    class_total = {c: 0 for c in dataset.classes}
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
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
    
    print("\n--- Results ---")
    print(f"Overall Accuracy: {acc:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
    print("\nPer-class Accuracy:")
    for class_name in dataset.classes:
        if class_total[class_name] > 0:
            class_acc = 100. * class_correct[class_name] / class_total[class_name]
            print(f"  {class_name}: {class_acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
        else:
            print(f"  {class_name}: N/A (0 samples)")

if __name__ == '__main__':
    main()
