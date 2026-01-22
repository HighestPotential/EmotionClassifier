import os
import numpy as np
import pandas as pd
from PIL import Image



class CustomDatasetLoaderwithoutResizing:

    def __init__(
        self,
        csv_path: str = r"C:\Users\Bax\Documents\cvdl\project\dataset.csv",
        img_path: str = r"C:\Users\Bax\Documents\cvdl\project\datasetsFaces"
        
        ):

        current_path = os.getcwd()
        self.csv_path = os.path.join(current_path, csv_path)
        self.img_path = os.path.join(current_path, img_path)

        
        
    def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:

        
        images = []

        df = pd.read_csv(self.csv_path)
        df = df.sample(len(df))

        labels = df.iloc[:, 1]
        filepaths = df.iloc[:, 0]

        for file in filepaths:
                        
            image = Image.open(file)
            image_arr = np.array(image).astype(np.uint8)

            images.append(image_arr)
        
        labels = labels.to_numpy()
        images = np.array(images)

        return images, labels