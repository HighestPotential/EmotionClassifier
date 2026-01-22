import os
import pandas as pd
import numpy as np
from PIL import Image

root_dir = "datasetsFaces"   # your main folder

label_map = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry"
}


rows = []

for label in sorted(os.listdir(root_dir)):
    label_path = os.path.join(root_dir, label)

    # skip files, only folders
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(label_path, file)
            rows.append([
                img_path,
                int(label)   # folder name is the label
            ])


print(rows[2].shape)

# print(type(rows))
df =pd.DataFrame(rows, columns=["path", "label"])
df.to_csv("datasets.csv", index= False)



