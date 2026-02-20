import os

from enum import Enum
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class DataMode(Enum):
    train: str = "train"
    eval:  str = "eval"
    test:  str = "test"

@dataclass
class DataSample:
    image: str
    label:str

class FERDataset(Dataset):
    def __init__(self, basedir: str, mode: DataMode, transform = None, target_transform = None):
        self.emotionMap = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 4,
            "surprise": 5,
        }
        
        self.data: list[DataSample] = []
        self.mode = mode.value

        self.transform = transform
        self.target_transform = target_transform

        for dataset in os.listdir(basedir):
            datasetPath: str = os.path.join(basedir, dataset, self.mode)
            
            for emotion in os.listdir(datasetPath):
                emotionPath = os.path.join(datasetPath, emotion)

                for image in os.listdir(emotionPath):
                    imgPath = os.path.join(emotionPath, image)
                    sample = DataSample(image=imgPath, label=emotion)

                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = np.array(Image.open(sample.image)).astype(np.uint8)
        label = self.emotionMap[sample.label]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label