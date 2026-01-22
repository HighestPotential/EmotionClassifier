import cv2
import numpy as np
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

<<<<<<< HEAD
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
=======


class resizing_to_64(ImageProcessor):
    """
    A class that resizes images to 64x64 pixels from a set of images. 
    For smaller images a cubic interpolation method has to be used.
    Experiments with the use from SRGAN will be executed.

    Attributes
    -----------

    None

    Methods
    --------

    process(self, image: np.ndarray, threshold = 100.0) -> np.ndarray
        processes n images stored in a numpy array of size (n, h, w) where h and 
        w are the height and width of the image
    """
      
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the image to the aimed 64x64 size. 
        If the image is smaller than 64x64, it uses cubic interpolation to upscale the image.
        

        :param image: supports (H,W) and (H,W,C) shape



        :type image: expects a np.ndarray


        :return: Returns None when the input "image" is None or empty 
                Otherwise it returns the 64x64 image


        :rtype: ndarray
        """

        resized_image = image.copy()

        if image is None:
                raise SkipImage("Input image is None")
        
        elif resized_image.size < 64:
            return cv2.resize(resized_image, (64,64), interpolation=cv2.INTER_CUBIC) 
        else:    
            return cv2.resize(resized_image, (64,64))
    
 
>>>>>>> main
