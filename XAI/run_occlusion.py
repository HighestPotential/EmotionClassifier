import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

IMG_SIZE = 64
SAVE_VIS_SIZE = 256

LIMIT_IMAGES = 24
SPLIT = "test"

PATCH = 8
STRIDE = 4
OCCLUDE_VALUE = 0.5 

SEED = 42

HARDCODED_WEIGHTS = Path(
    r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\Best Results ResNet18_SE\best_resnet18_se_12.01.26.pth"
)

HARDCODED_PROJECT_ROOT = Path(
    r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks"
)


def find_project_root():
    if HARDCODED_PROJECT_ROOT.exists():
        return HARDCODED_PROJECT_ROOT
    here = Path(__file__).resolve().parent
    return here


def find_ready_root(project_root: Path):
    candidates = [
        project_root / "ready_to_use_datasets",
        Path(
            r"C:\Users\drnes\OneDrive\Desktop\PC\Aleks\Aleks Uni\Computer Vision\Final_Project\EmotionClassifier\EmotionClassifier-aleks\ready_to_use_datasets"
        ),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError("Could not find ready_to_use_datasets.")


def list_images(ready_root: Path, split=SPLIT, limit=LIMIT_IMAGES):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = []
    for ds_dir in sorted(ready_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        split_dir = ds_dir / split
        if not split_dir.is_dir():
            continue
        for emo in CLASSES:
            emo_dir = split_dir / emo
            if not emo_dir.is_dir():
                continue
            for p in emo_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    paths.append((p, emo, ds_dir.name))
    random.shuffle(paths)
    return paths[:limit]


def load_model(weights_path: Path, device):
    from models.aleks.aleks_resnet18_se import ResNet18SE

    model = ResNet18SE(num_classes=6).to(device)

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)

    if any(k.startswith("fc.1.") for k in state.keys()):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 6)).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("Warning: load_state_dict mismatch")
        if missing:
            print("Missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
        if unexpected:
            print("Unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

    model.eval()
    return model


@torch.no_grad()
def predict_probs(model, x):
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    return probs, logits


def occlusion_sensitivity(model, x, class_idx, patch=PATCH, stride=STRIDE, occlude_value=OCCLUDE_VALUE):

    _, _, H, W = x.shape
    base_probs, _ = predict_probs(model, x)
    base_p = float(base_probs[0, class_idx].item())

    heat = torch.zeros((H, W), dtype=torch.float32, device=x.device)
    cnt = torch.zeros((H, W), dtype=torch.float32, device=x.device)

    for y in range(0, H - patch + 1, stride):
        for z in range(0, W - patch + 1, stride):
            x_occ = x.clone()
            x_occ[:, :, y : y + patch, z : z + patch] = occlude_value
            probs, _ = predict_probs(model, x_occ)
            p = float(probs[0, class_idx].item())
            
            delta = base_p - p
            
            heat[y : y + patch, z : z + patch] += delta
            cnt[y : y + patch, z : z + patch] += 1.0

    heat = heat / torch.clamp(cnt, min=1.0)

    heat_np = heat.detach().cpu().numpy()
    
    heat_np = np.maximum(heat_np, 0)
    
    heat_max = heat_np.max()
    if heat_max > 1e-6:
        heat_np = heat_np / heat_max
    
    return heat_np, base_p


def overlay_heatmap(img_pil: Image.Image, heat_2d: np.ndarray, max_alpha=0.65):

    img_rgba = img_pil.convert("RGBA")
    w, h = img_rgba.size


    heat_resized = cv2.resize(heat_2d, (w, h), interpolation=cv2.INTER_CUBIC)

    kernel_size = max(15, min(w, h) // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    heat_blurred = cv2.GaussianBlur(heat_resized, (kernel_size, kernel_size), 0)
    
    heat_norm = np.clip(heat_blurred, 0, 1)
    

    gamma = 1.5
    heat_norm = np.power(heat_norm, 1/gamma)
    
    heat_uint8 = np.uint8(255 * heat_norm)

    heatmap_color_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    alpha_values = np.power(heat_norm, 2.0) * max_alpha
    alpha_mask = np.uint8(255 * alpha_values)


    heatmap_pil = Image.fromarray(heatmap_color_rgb)
    mask_pil = Image.fromarray(alpha_mask, mode='L')
    heatmap_pil.putalpha(mask_pil)

    img_rgba.alpha_composite(heatmap_pil)
    return img_rgba.convert("RGB")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    project_root = find_project_root()
    out_dir = project_root / "occlusion_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    ready_root = find_ready_root(project_root)
    samples = list_images(ready_root, split=SPLIT, limit=LIMIT_IMAGES)
    if not samples:
        raise RuntimeError(f"No images found under: {ready_root}")

    weights_path = HARDCODED_WEIGHTS if HARDCODED_WEIGHTS.exists() else (project_root / "best_resnet18_se.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found. Tried:\n  {HARDCODED_WEIGHTS}\n  {project_root / 'best_resnet18_se.pth'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(weights_path, device)

    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    print(f"Processing {len(samples)} images...")
    for i, (img_path, gt, ds_name) in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] Processing {img_path.name}...")
        
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        probs, _ = predict_probs(model, x)
        pred_idx = int(probs.argmax(1).item())
        pred = IDX_TO_CLASS[pred_idx]
        conf = float(probs[0, pred_idx].item())

        heat, base_p = occlusion_sensitivity(
            model, x, pred_idx, 
            patch=PATCH, 
            stride=STRIDE, 
            occlude_value=OCCLUDE_VALUE
        )
        
        vis_img = img.resize((SAVE_VIS_SIZE, SAVE_VIS_SIZE), Image.BICUBIC)
        overlay = overlay_heatmap(vis_img, heat, max_alpha=0.65)

        safe_stem = img_path.stem.replace(" ", "_")[:80]
        out_name = f"{i:02d}__ds-{ds_name}__gt-{gt}__pred-{pred}__conf-{conf:.2f}__patch{PATCH}_s{STRIDE}__{safe_stem}.png"
        overlay.save(out_dir / out_name)
        
        print(f"  GT: {gt} | Pred: {pred} ({conf:.2%}) | Heat range: [{heat.min():.3f}, {heat.max():.3f}]")

    print(f"\n Saved {len(samples)} occlusion maps to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()