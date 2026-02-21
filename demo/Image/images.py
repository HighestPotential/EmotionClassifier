# |===================================================================================================|
# |Usage:                                                                                             |
# |===================================================================================================|
# |python3 main.py [in_dir] [out_file]                                                                |
# |                                                                                                   |
# |in_dir is the directory where all the images to evaluate are located                               |
# |out_file is the final csv file where filepaths and respective probabilites for ech class are stored|
# |===================================================================================================|


import os
import argparse
import csv
import math

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from aleks_resnet18_se import ResNet18SE

class EvalDataset(Dataset):
    def __init__(self, root: str, transform = None):
        self.images = []
        self.transform = transform

        for img in os.listdir(root):
            filePath = os.path.join(root, img)
            self.images.append(filePath)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path)
        img = np.array(img).astype(np.uint8)

        if self.transform:
            img = self.transform(img)

        return path, img
    
def formatProbs(probs: list[list[float]]):
    result = []
    for l in probs:
        l = [math.trunc(x * 100) / 100 for x in l]
        l = [f"{x:.2f}" for x in l]
        result.append(l)
    
    return result

def main(input: str, output: str) -> None:
    output_headers = ["Filepath", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

    model = ResNet18SE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("ResNet18_trained.pth", weights_only=False, map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = EvalDataset(input, transform=transform)
    loader = DataLoader(dataset, batch_size=64)
    softmax = nn.Softmax(dim=1)

    with open(output, "w") as file:
        writer = csv.writer(file)
        writer.writerow(output_headers)

        for paths, img in loader:
            img.to(device)
            logits = model(img)
            probs = softmax(logits)

            pathList = [*paths]
            dataList = probs.tolist()
            dataList = formatProbs(dataList)

            outData = list(map(lambda x, y: [x] + y, pathList, dataList))
            writer.writerows(outData)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Input directory containing images to evaluate")
    parser.add_argument("-of", help="Output csv file", default="./result.csv", required=False)

    args = parser.parse_args()

    dataDir = args.in_dir
    output = args.of

    main(dataDir, output)