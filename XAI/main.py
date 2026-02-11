import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aleks_resnet18_se import ResNet18SE

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

IMG_SIZE = 64
SAVE_VIS_SIZE = 256
LIMIT_IMAGES = 24
SPLIT = "test"

WEIGHTS_PATH = PROJECT_ROOT / "models" / "aleks" / "weights" / "best_resnet18_se_SGD_cbfl.pth"
READY_ROOT = PROJECT_ROOT / "ready_to_use_datasets"
OUT_DIR = PROJECT_ROOT / "gradcam_out"

class GradCAM(nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.target = target_layer

        self.gradients = None

    def forward_hook(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_activations(self, x):
        return self.model(x)

    def forward():
        pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18SE()
    model.load_state_dict(torch.load("./ResNet18_trained.pth", map_location=device, weights_only=False))

    print(model)

    target_layer = model.layer4[1]
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
