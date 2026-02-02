import sys
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from aleks_resnet18_se import ResNet18SE

class ImageLoader(Dataset):
    def __init__(self, opRoot: str):
        self.data: list[str] = []
        fullPath = os.path.join(os.getcwd(), opRoot)

        for file in os.listdir(fullPath):
            filePath = os.path.join(fullPath, file)
            self.data.append(filePath)
    
    def get_images(self):
        return self.data

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        return image, None

    def __len__(self) -> int:
        return len(self.data)
    
    def checkDir(self, path: str) -> bool:
        return os.path.isdir(path)

class GradCAM(nn.Module):
    def __init__(self, weights: str, device: torch.device):
        super().__init__()

        self.model = ResNet18SE()
        self.model.load_state_dict(torch.load(weights, map_location=device, weights_only=False))
        self.model.to(device)

        self.conv = nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu,
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4
                )

        self.pool = self.model.pool
        self.classifier = self.model.fc 

        self.gradients = None

    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv(x)

        h = x.register_hook(self.activation_hook)

        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.conv(x)

def generate_Map(model: GradCAM, device: torch.device, image: torch.Tensor, loader: DataLoader):
    model.eval()
    image = image.to(device).unsqueeze(0)

    for img, _ in loader:
    pred = model(image)

    model.zero_grad()
    pred[:, 0].backward()

    gradients = model.get_activations_gradient()
    activations = model.get_activations(image).detach()

    pooled = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)

    heatmap /= torch.max(heatmap)
    return heatmap

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

if __name__ == "__main__":
    weights = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam = GradCAM(weights, device)

    loader = DataLoader(ImageLoader("/home/daniel/Pictures/CKPlusIm"))
