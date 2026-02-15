import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

from aleks_resnet18_se import ResNet18SE

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18SE()
model = model.to(dev)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

dataset = datasets.ImageFolder(root='~/Pictures/CKplusIm', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

for batch_idx, (input_tensor, labels) in enumerate(dataloader):
    input_tensor = input_tensor.to(dev)
    
    targets = [ClassifierOutputTarget(label.item()) for label in labels]
    
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    
    for i in range(input_tensor.size(0)):
        img = input_tensor[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406] # Un-normalize
        img = np.clip(img, 0, 1)
        
        vis = show_cam_on_image(img, grayscale_cams[i, :], use_rgb=True, image_weight=0.8)
        cv2.imwrite(f'results/cam_{batch_idx}_{i}.jpg', vis)