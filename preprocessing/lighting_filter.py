import numpy as np
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class LightingFilter(ImageProcessor):

    def __init__(self, low_mean: float = 20.0, high_mean: float = 235.0, min_std: float = 12.0, max_black_frac: float = 0.60, max_white_frac: float = 0.60, black_thresh: int = 5, white_thresh: int = 250):
        self.low_mean = low_mean
        self.high_mean = high_mean
        self.min_std = min_std
        self.max_black_frac = max_black_frac
        self.max_white_frac = max_white_frac
        self.black_thresh = black_thresh
        self.white_thresh = white_thresh

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise SkipImage("Image is None")
        
        if image.ndim == 2:
            gray = image.astype(np.float32)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            gray = image[:, :, :3].astype(np.float32).mean(axis = 2)
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0].astype(np.float32)
        else:
            raise ValueError(f"Unsupported image shape for lighting filter: {image.shape}")
        
        if np.issubdtype(gray.dtype, np.floating) and gray.max() <= 1.0:
            gray = gray * 255.0
        
        mean = float(gray.mean())
        std = float(gray.std())
        black_frac = float((gray <= self.black_thresh).mean())
        white_frac = float((gray >= self.white_thresh).mean())

        if mean < self.low_mean:
            raise SkipImage(f"Too dark (mean={mean:.1f})")
        if mean > self.high_mean:
            raise SkipImage(f"Too bright (mean={mean:.1f})")
        if std < self.min_std:
            raise SkipImage(f"Too low contrast (std={std:.1f})")
        if black_frac > self.max_black_frac:
            raise SkipImage(f"Too many dark pixels (black_frac={black_frac:.2f})")
        if white_frac > self.max_white_frac:
            raise SkipImage(f"Too many bright pixels (white_frac={white_frac:.2f})")

        return image