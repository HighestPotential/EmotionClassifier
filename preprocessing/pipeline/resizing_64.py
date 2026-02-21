import cv2
import numpy as np
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class ResizingTo64(ImageProcessor):
    # CHANGED: Imported the descriptive docstring from the main branch.
    """
    A class that resizes images to 64x64 pixels while preserving aspect ratio.
    For smaller images, a cubic interpolation method is utilized.
    """
      
    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise SkipImage("Input image is None")
        
        if image.size == 0:
            raise SkipImage("Input image is Empty")
            
        # 1. Calculate how much padding is needed to make it square
        h, w = image.shape[:2]
        
        longest_side = max(h, w)
        
        # Calculate padding amounts
        top = (longest_side - h) // 2
        bottom = longest_side - h - top
        left = (longest_side - w) // 2
        right = longest_side - w - left
        
        # 2. Add black borders (padding)
        square_image = cv2.copyMakeBorder(
            image, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # 3. Resize the now-square image to 64x64
        if square_image.shape[0] < 64:

             return cv2.resize(square_image, (64, 64), interpolation=cv2.INTER_CUBIC)
        else:

             return cv2.resize(square_image, (64, 64), interpolation=cv2.INTER_AREA)
