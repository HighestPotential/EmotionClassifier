import numpy as np
from load_dataset import CustomDatasetLoader
import matplotlib.pyplot as plt

import cv2 as cv

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "suprise",
}

MAXIMUM = 2000

loader = CustomDatasetLoader()
images, labels = loader.load_dataset(max_samples=MAXIMUM)

for i in range(10):
    ix = np.random.randint(low = 0, high=MAXIMUM - 1)
    image = images[ix]
    label = labels[ix]

    variance = cv.Laplacian(image, cv.CV_64F).var()

    ax = plt.subplot(2, 5, i + 1)
    ax.imshow(image, cmap="gray")
    ax.set_title(f"var: {variance:.1f}")
    ax.set_xlabel(f"{emotion_map[label]}")
    ax.axis("off")


plt.show()

unique_labels, label_count = np.unique(labels, return_counts=True)
plt.bar(unique_labels, label_count)
plt.show()