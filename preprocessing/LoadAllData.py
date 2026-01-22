import os

import numpy as np
from PIL import Image

from image_duplicates_old import RemoveDuplicates

class ImageLoader:
    def __init__(self, opRoot: str):
        self.data: list[str] = []
        fullPath = os.path.join(os.getcwd(), opRoot)

        for dataset in os.listdir(fullPath):
            currentDataset: str = os.path.join(fullPath, dataset)
            if not self.checkDir(currentDataset):
                continue


            for split in os.listdir(currentDataset):
                currentSplit = os.path.join(currentDataset, split)
                if not self.checkDir(currentSplit):
                    continue

                for emotion in os.listdir(currentSplit):
                    currentEmotion = os.path.join(currentSplit, emotion)

                    if not self.checkDir(currentEmotion):
                        continue

                    for file in os.listdir(currentEmotion):
                        filePath = os.path.join(currentEmotion, file)
                        self.data.append(filePath)


    def __len__(self) -> int:
        return len(self.data)
    
    def checkDir(self, path: str) -> bool:
        return os.path.isdir(path)

    def handle(self):
        images = []

        for p in self.data:
            imgArr = np.array(Image.open(p)).astype(np.uint8)
            imgArr = imgArr.mean(-1)
            images.append(imgArr)
        
        remover = RemoveDuplicates()
        mask = remover.process(np.array(images))

        duplicatePaths = [f for f in self.data if mask[self.data.index(f)] == 0]
        for dup in duplicatePaths:
            os.remove(dup)


if __name__ == "__main__":
    l = ImageLoader("../ready_to_use_datasets")
    l.handle()