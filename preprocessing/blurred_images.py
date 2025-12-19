import numpy as np
import cv2 as cv
from image_processor_interface import ImageProcessor

class RemoveBlurredFaces(ImageProcessor):

    """
    A class that removes blurry images from a set of images

    Attributes:
    -----------

    None

    Methods:
    --------

    process(self, image: np.ndarray, threshold = 100.0) -> np.ndarray
        processes n images stored in a numpy array of size (n, h, w) where h and 
        w are the height and width of the image
    """

    def process(self, image: np.ndarray, threshold = 100.0) -> np.ndarray:
        """
        Removes items from a list of images according to the class description

        Parameters
        -----------

        image: np.ndarray
            A numpy array that stores images. The shape of the array is expected to be (n, h, w)
            where:
                - n: number of images
                - h: height of each image
                - w: width of each image
        
        threshold: float, optional
            The threshold that classifies an image as blurry. If the laplaian variance < threshold the image 
            is considered blurry and removed.

        Returns
        --------

        np.ndarray
            A copy of the original image array where the blurry images were removed.

        """

        cleaned_array = image.copy()
        
        variances = [cv.Laplacian(sample, cv.CV_64F).var() for sample in cleaned_array]
        variances = np.array(variances)

        mask = variances >= threshold
        cleaned_array = cleaned_array[mask]

        return cleaned_array