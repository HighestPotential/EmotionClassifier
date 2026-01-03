import numpy as np
import cv2 as cv
from skip_image import SkipImage
from image_processor_interface import ImageProcessor

class RemoveBlurredFaces(ImageProcessor):

    """
    A class that removes blurry images from a set of images

    Attributes
    -----------

    None

    Methods
    --------

    process(self, image: np.ndarray, threshold = 100.0) -> np.ndarray
        processes n images stored in a numpy array of size (n, h, w) where h and 
        w are the height and width of the image
    """

    def process(self, image: np.ndarray, threshold = 100.0) -> np.ndarray: # chaged bacause it expects multiple images
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
        
        labels: np.ndarray
            A numpy array that stores the labels to the images. The expected size is (n) or (n, 1).
            Both variations produce identical results.
        
        threshold: float, optional
            The threshold that classifies an image as blurry. If the laplaian variance < threshold the image 
            is considered blurry and removed.

        Returns
        --------

        cleaned_images: np.ndarray
            A copy of the original image array where the blurry images were removed.

        cleaned_labels: np.ndarray
            A copy of the original label array where the blurry image labels were removed
        """
        
        if image is None:
            raise SkipImage("Image is None")

        # Convert to grayscale for Laplacian
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate Laplacian variance
        variance = cv.Laplacian(gray, cv.CV_64F).var()

        if variance < threshold:
            raise SkipImage(f"Image is blurry (variance={variance:.2f} < {threshold})")

        return image
