import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv

class CustomDatasetLoader:
    def __init__(self,
        csv_path: str = "format_datasets/dataset.csv",
        img_path: str = "format_datasets/dataset/"
        ) -> None:
        current_path = os.getcwd()
        self.csv_path = os.path.join(current_path, csv_path)
        self.img_path = os.path.join(current_path, img_path)
    
    def load_dataset(self, max_samples: int = None) -> tuple[np.ndarray, np.ndarray]:
        images = []

        df = pd.read_csv(self.csv_path)
        df = df.sample(max_samples if max_samples is not None else len(df))

        labels = df.iloc[:, 0]
        filenames = df.iloc[:, 1]

        for file in filenames:
            filepath = os.path.join(self.img_path, file)
            
            image = Image.open(filepath)
            image_arr = np.array(image).astype(np.uint8)

            if image_arr.ndim > 2:
                image_arr = image_arr.mean(axis=-1)

            image_arr = cv.resize(image_arr, (150, 150),cv.INTER_CUBIC)
            images.append(image_arr)
        
        labels = labels.to_numpy()
        images = np.array(images)

        return images, labels
