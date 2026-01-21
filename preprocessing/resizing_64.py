import cv2
import numpy as np
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class ResizingTo64(ImageProcessor):
    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise SkipImage("Input image is None")
            
        # 1. Calculate how much padding is needed to make it square
        h, w = image.shape[:2]
        
        # Determine the target size (the largest side)
        longest_side = max(h, w)
        
        # Calculate padding amounts
        top = (longest_side - h) // 2
        bottom = longest_side - h - top
        left = (longest_side - w) // 2
        right = longest_side - w - left
        
        # 2. Add black borders (padding)
        # BORDER_CONSTANT adds a solid color (0,0,0 is black)
        square_image = cv2.copyMakeBorder(
            image, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # 3. Resize the now-square image to 64x64
        # Now it shrinks equally because the input is already a square
        if square_image.shape[0] < 64:
             return cv2.resize(square_image, (64, 64), interpolation=cv2.INTER_CUBIC)
        else:
             return cv2.resize(square_image, (64, 64), interpolation=cv2.INTER_AREA)
