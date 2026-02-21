from abc import ABC, abstractmethod
import numpy as np


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the input image and return the processed image.
        """
        pass
