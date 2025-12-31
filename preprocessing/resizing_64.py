import cv2
import numpy as np
from image_processor_interface import ImageProcessor
from skip_image import SkipImage



class ResizingTo64(ImageProcessor):
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
        if image is None:
            raise SkipImage("Input image is None")
        
        if image.size == 0:
            raise SkipImage("Input image is Empty")
        
        resized_image = image.copy()
        
        if resized_image.size < 64:
            return cv2.resize(resized_image, (64,64), interpolation=cv2.INTER_CUBIC) 
        else:    
            return cv2.resize(resized_image, (64,64))
    
 