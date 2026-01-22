import os
from find_duplicates import FindDuplicates

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
    
    def get_images(self):
        return self.data


    def __len__(self) -> int:
        return len(self.data)
    
    def checkDir(self, path: str) -> bool:
        return os.path.isdir(path)

if __name__ == "__main__":
    imageRoot = "../testData"
    handler = FindDuplicates()

    images = ImageLoader(os.path.abspath(imageRoot)).get_images()
    duplicates = handler.find_duplicates(images)
    
    print(f"Total images: {len(images)}")
    print(f"Duplicates found: {len(duplicates)}")