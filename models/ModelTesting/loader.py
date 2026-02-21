import os

from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class DataSample:
    image: str
    label:str

class FERDataset(Dataset):
    def __init__(self, basedir: str, transform = None, target_transform = None):
        self.emotionMap = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 4,
            "surprise": 5,
        }
        
        self.data: list[DataSample] = []
        self.rootDir = basedir
        self.transform = transform
        self.target_transform = target_transform

    def loadDataset(self, wanted: str = ""):
        for dataset in os.listdir(self.rootDir):
            if not wanted in dataset:
                continue
            
            datasetPath = os.path.join(self.rootDir, dataset)
        
            for split in os.listdir(datasetPath):
                currentSplit = os.path.join(datasetPath, split)

                for emotion in os.listdir(currentSplit):

                    currentEmotion = os.path.join(currentSplit, emotion)
                    for img in os.listdir(currentEmotion):
                        imgPath = os.path.join(currentEmotion, img)
                        self.data.append(DataSample(image=imgPath, label=emotion))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = Image.open(sample.image)
        label = self.emotionMap[sample.label]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
