import numpy as np
from image_processor_interface import ImageProcessor

class ToRGB(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image is None")

        # HxW grayscale
        if image.ndim == 2:
            return np.repeat(image[:, :, None], 3, axis=2)

        # HxWxC
        if image.ndim == 3:
            c = image.shape[2]

            if c == 1:  # HxWx1 grayscale
                return np.repeat(image, 3, axis=2)

            if c == 4:  # RGBA -> RGB
                return image[:, :, :3]

            if c == 3:  # already RGB
                return image

        raise ValueError(f"Unsupported image format: shape={image.shape}")
