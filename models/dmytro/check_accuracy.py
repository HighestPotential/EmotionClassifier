
import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import timm

# --- Path Setup ---
# Needed for Vim and CCT imports
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path for CCT
cct_path = os.path.join(script_dir, 'CCT-7withSAM')
if cct_path not in sys.path:
    sys.path.append(cct_path)
# Path for Vim
vim_path = os.path.join(script_dir, 'VIM/Vim')
if vim_path not in sys.path:
    sys.path.append(vim_path)
    sys.path.append(os.path.join(vim_path, 'vim'))

# --- Constants ---
IMG_SIZE = 64
BATCH_SIZE = 128
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
MEAN = [0.4681, 0.4447, 0.4560]
STD = [0.2327, 0.2227, 0.2224]

# --- Model Factories ---

def create_vim(num_classes):
    try:
        from vim.models_mamba import VisionMamba
        model = VisionMamba(
            patch_size=8,
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
            if_devide_out=True,
            use_middle_cls_token=True,
            img_size=IMG_SIZE,
            num_classes=num_classes,
            stride=8
        )
        return model
    except ImportError:
        import traceback
        traceback.print_exc()
        print("Error: Could not import VisionMamba. Ensure 'vim' package is in path.")
        sys.exit(1)

def create_effnet(num_classes):
    print("Creating EfficientNetV2-S (Overfitting Fix Config)...")
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=num_classes)
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

def create_convnext(num_classes):
    print("Creating ConvNeXt V2 Atto (Overfitting Fix Config)...")
    model = timm.create_model('convnextv2_atto', pretrained=False, num_classes=num_classes)
    # Stem Fix
    model.stem = nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1)
    return model

def create_cct(num_classes):
    try:
        from custom_cct import CCT
        model = CCT(
            img_size=IMG_SIZE,
            embedding_dim=256,
            n_input_channels=3,
            n_conv_layers=1,
            kernel_size=3,
            stride=1,
            padding=1,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_layers=7,
            num_heads=4,
            mlp_ratio=2.0,
            num_classes=num_classes,
            positional_embedding='learnable'
        )
        return model
    except ImportError:
        print("Error: Could not import CCT. Ensure 'custom_cct.py' is in path.")
        sys.exit(1)

# --- Data Loading ---
def get_dataloader(data_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    if os.path.exists(data_path):
        ds = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True), len(ds)
    else:
        return None, 0

# --- Evaluation Loop ---
def evaluate(model, loader, device, name="Dataset"):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {name}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item() * inputs.size(0)
            
    acc = 100. * correct / total
    avg_loss = running_loss / total
    print(f"[{name}] Accuracy: {acc:.2f}% | Loss: {avg_loss:.4f}")
    return acc

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Accuracy of Emotion Models")
    parser.add_argument('--model', type=str, required=True, choices=['vim', 'effnet', 'convnext', 'cct'], help='Model type to load')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint file')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Create Model
    if args.model == 'vim':
        model = create_vim(len(CLASSES))
    elif args.model == 'effnet':
        model = create_effnet(len(CLASSES))
    elif args.model == 'convnext':
        model = create_convnext(len(CLASSES))
    elif args.model == 'cct':
        model = create_cct(len(CLASSES))
    
    model = model.to(device)
    
    # 2. Load Checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_ema_state_dict' in checkpoint:
            print("Loaded EMA state dict (Best Practice for inference)")
            state_dict = checkpoint['model_ema_state_dict']
            # Sometimes EMA dict keys have "module.", check and strip
            # But usually ModelEmaV2 stores it cleanly or as is.
            # timm ModelEmaV2 state dict usually matches model.
        elif 'model_state_dict' in checkpoint:
            print("Loaded standard model_state_dict")
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume raw state dict
            state_dict = checkpoint
            
        # Load weights
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"Weights loaded: {msg}")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
        
    # 3. Data Loading
    # Check all datasets
    dataset_root = '/home/d/dumanskyy/work/EmotionClassifier/latest_2_0_ready_to_use_datasets'
    datasets_to_check = [
        'AffectNet', 'CKplusIm', 'FERPlus', 'RAF-DB', 
        'KDEFFormated', 'jaffeFormated', 'MMAFEDB', 
        'NONAMEFormated', 'ExpWFormated', 'EmoSet-118k',
        'NHFI', 'WSEFEPFormated'
    ]
    
    for ds_name in datasets_to_check:
        print(f"\n--- Checking {ds_name} ---")
        ds_path = os.path.join(dataset_root, ds_name)
        
        # Val
        val_loader, val_len = get_dataloader(os.path.join(ds_path, 'eval'))
        if val_loader:
             evaluate(model, val_loader, device, name=f"{ds_name} [Val]")
        else:
            print(f"Skipping {ds_name} Val (Not found)")
            
        # Test
        test_loader, test_len = get_dataloader(os.path.join(ds_path, 'test'))
        if test_loader:
            evaluate(model, test_loader, device, name=f"{ds_name} [Test]")
        else:

            pass 
