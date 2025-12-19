import numpy as np
import cv2 as cv
from image_processor_interface import ImageProcessor

class RemoveDuplicates(ImageProcessor):
    
    def difference_hash(self, image: np.ndarray, hashsize: int = 8) -> int:
        resized: np.ndarray = cv.resize(image, (hashsize + 1, hashsize))
        
        differences: np.ndarray = resized[:, 1:] > resized[:, :-1]
        differences = differences.flatten()

        hash = sum([2**i for (i, v) in enumerate(differences) if v])
        return hash

    def process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError