import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from blurred_images import RemoveBlurredFaces

handler = RemoveBlurredFaces()
test_dir: str = "/home/daniel/Documents/University/WS_25_26/Softwarepraktikum/Projekt/EmotionClassifier/datasets/RAF-DB/train/2/"

images = []

for file in os.listdir(test_dir):
    path = os.path.join(test_dir, file)
    img = Image.open(path)
    img = np.array(img)
    img = np.mean(img, axis=-1)
    images.append(img)

images = np.array(images)
n_old, _, _ = images.shape
accepted = handler.process(images)

n, _, _ = accepted.shape
print(f"Accepted images: {n}/{n_old}, about {100 * n / n_old}%")

if n < 1:
    raise ValueError

plt.imshow(accepted[0])
plt.show()