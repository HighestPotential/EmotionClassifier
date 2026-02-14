import sys
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aleks.aleks_resnet18_se import ResNet18SE

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

IMG_SIZE = 64
SAVE_VIS_SIZE = 256
LIMIT_IMAGES = 24
SPLIT = "test"
SEED = 42

WEIGHTS_PATH = PROJECT_ROOT / "models" / "aleks" / "weights" / "best_resnet18_se_SGD_cbfl.pth"
READY_ROOT = PROJECT_ROOT / "ready_to_use_datasets"
OUT_DIR = PROJECT_ROOT / "gradcam_out"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root, split, limit):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = []
    for ds in sorted(root.iterdir()):
        if not ds.is_dir():
            continue
        split_dir = ds / split
        if not split_dir.is_dir():
            continue
        for emo in CLASSES:
            emo_dir = split_dir / emo
            if not emo_dir.is_dir():
                continue
            for p in emo_dir.rglob("*"):
                if p.suffix.lower() in exts:
                    paths.append((p, emo))
    random.shuffle(paths)
    return paths[:limit]


def load_model(weights_path, device):
    model = ResNet18SE(num_classes=6).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)

    if any(k.startswith("fc.1.") for k in state.keys()):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 6)
        ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x):
        self.model.zero_grad()

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam[0].cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        confidence = probs[0, class_idx].item()
        return cam, class_idx, confidence


def overlay(img, cam, alpha=0.6):
    w, h = img.size
    cam = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * np.array(img) + alpha * heatmap
    return Image.fromarray(np.uint8(blended))


def make_grid(img, cam):
    img = img.resize((SAVE_VIS_SIZE, SAVE_VIS_SIZE))
    overlay_img = overlay(img, cam)

    cam_gray = np.uint8(255 * cv2.resize(cam, (SAVE_VIS_SIZE, SAVE_VIS_SIZE)))
    cam_color = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)
    cam_color = Image.fromarray(cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB))

    grid = Image.new("RGB", (SAVE_VIS_SIZE * 3, SAVE_VIS_SIZE))
    grid.paste(img, (0, 0))
    grid.paste(overlay_img, (SAVE_VIS_SIZE, 0))
    grid.paste(cam_color, (SAVE_VIS_SIZE * 2, 0))
    return grid


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = list_images(READY_ROOT, SPLIT, LIMIT_IMAGES)
    if not samples:
        raise RuntimeError("No images found.")

    model = load_model(WEIGHTS_PATH, device)
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for i, (img_path, gt) in enumerate(samples, 1):
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        cam, pred_idx, conf = gradcam(x)
        pred = IDX_TO_CLASS[pred_idx]

        grid = make_grid(img, cam)
        name = f"{i:02d}_gt-{gt}_pred-{pred}_conf-{conf:.2f}.png"
        grid.save(OUT_DIR / name)

if __name__ == "__main__":
    main()
