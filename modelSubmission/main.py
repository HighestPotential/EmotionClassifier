import os
import argparse
from pathlib import Path
import csv

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from aleks_resnet18_se import ResNet18SE

def loadImages(paths: list[str])-> np.ndarray:
    images = []

    for path in paths:
        img = Image.open(path)
        images.append(np.array(img).astype(np.uint8))
    
    return images

class FolderImages:
    def __init__(self, folder: str, transform):
        self.folder = Path(folder)
        self.transoform = transform

        self.paths = [p for p in self.folder.rglob("*") if p.is_file()]


def main(input: str, output: str) -> None:
    output_headers = ["filepath", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    model = ResNet18SE()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("ResNet18_trained.pth", weights_only=False, map_location=device))

    # Load Files
    paths = []
    fileDir = os.path.join(os.getcwd(), input)
    files = os.listdir(fileDir)
    for file in files:
        filePath = os.path.join(fileDir, file)
        paths.append(filePath)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data = loadImages(paths=paths)
    dataTensor = torch.tensor(np.array([transform(dp) for dp in data]))

    logits = model(dataTensor)
    softmax = nn.Softmax(dim=1)
    probs = softmax(logits)

    all_probs = probs.tolist()
    for i in range(len(all_probs)):
        file = paths[i]
        result = all_probs[i]
        result = [f"{r:.3f}" for r in result]

        with open(output, mode="a") as f:
            writer = csv.writer(f)
            if i == 0:
                writer.writerow(output_headers)
            writer.writerow([file] + result)
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Input directory containing images to evaluate")
    parser.add_argument("out_file", help="Output csv file")

    args = parser.parse_args()

    dataDir = args.in_dir
    output = args.out_file

    main(dataDir, output)

    df = pd.read_csv(output)
    print(df)