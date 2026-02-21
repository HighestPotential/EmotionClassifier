import os
import sys
import argparse
from dataclasses import dataclass

from loader import FERDataset

from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

from structure2tryWithoutTestsCorrect import CERN

@dataclass
class CNNContext:
    model: nn.Module
    device: torch.device
    saveFile: str

def testAccuracy(ctx: CNNContext, test_loader: DataLoader) -> float:
    transform = transforms.RandomHorizontalFlip(1.0)
    ctx.model.load_state_dict(torch.load(ctx.saveFile, weights_only=False, map_location=ctx.device)["model_state_dict"])

    ctx.model.eval()
    with torch.no_grad():
        correct = 0
        for image, label in tqdm(test_loader, desc="Test progress", leave=False):
            image = image.to(ctx.device)
            label = label.to(ctx.device)
            image2 = transform(image)
            
            out1 = ctx.model(image, image2)

            out2 = ctx.model(image, image2)

            out = (out1 + out2) / 2

            correct += (out.argmax(1) == label).sum().item()
        
    accuracy = 100 * correct / len(test_loader.dataset)
    
    return accuracy

if __name__ == "__main__":
    normalTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    DATAPATH = os.path.join(os.getcwd(), "data")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", type=str, default="", help="dataset to load. Empty will load all")
    parser.add_argument("-bs", type=int, default=64, help="Batchsize to use for testing")
    parser.add_argument("-w", "--weights", type=str, default="model.pth", help="File containing the model weights", required=True)
    args = parser.parse_args()

    wantedDS = args.ds
    batchSize = args.bs
    weights = args.weights

    print("INFO")
    print("_______________")
    print(f"Dataset: {wantedDS if wantedDS else "all"}")
    print(f"Batchsize: {batchSize}")
    print(f"Weights: {weights}")
    print("_______________")

    normalLoader = FERDataset(DATAPATH, transform=normalTransform)
    normalLoader.loadDataset(wantedDS)

    if len(normalLoader) < 1:
        print("Error loading data")
        sys.exit(1)

    normalLoader = DataLoader(normalLoader, batch_size=batchSize, shuffle=False)

    #model = models.GoogLeNet(num_classes=6, aux_logits=False)
    model = CERN()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveFile = weights

    ctx = CNNContext(
            model=model,
            device=device,
            saveFile=saveFile
            )
    
    acc = testAccuracy(ctx, normalLoader)

    print(f"Accuracy: {acc}")
