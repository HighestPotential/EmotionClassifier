from image_processor_interface import ImageProcessor
import numpy as np

class ToRGB(ImageProcessor):
    
    def process(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:  # Grayscale image
            return np.stack((image,) * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA image
            return image[:, :, :3]
        elif image.shape[2] == 3:  # Already RGB
            return image
        else:
            raise ValueError("Unsupported image format")
    
    def new_function(self):
        return "ToRGB()"
