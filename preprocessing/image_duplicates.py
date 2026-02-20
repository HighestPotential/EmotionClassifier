import numpy as np
import cv2 as cv

from image_processor_interface import ImageProcessor

class RemoveDuplicates(ImageProcessor):

    """
    A class that removes duplicate images from a set of images

    Attributes
    -----------
    
    None

    Methods
    --------
    _difference_hash(self, image: np.ndarray, hashsize: int = 8) -> int:
        calculates the difference hash (dhash) of an image


    process(self, image: np.ndarray, threshold = 100.0) -> np.ndarray
        processes n images stored in a numpy array of size (n, h, w) where 
        - h: image height 
        - w: image width
    """

    def _difference_hash(self, image: np.ndarray, hashsize: int = 8) -> int:
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
        
        hashsize: int, optional
            The size the image will be shrunken to to compute the hash. With a 
            hashsize of n the resulting hash will be of length 8x8 Bits

        Returns
        --------

        int
            The computed distance hash
        """
        resized: np.ndarray = cv.resize(image, (hashsize + 1, hashsize))
        
        differences: np.ndarray = resized[:, 1:] > resized[:, :-1]
        differences = differences.flatten()

        hash = sum([2**i for (i, v) in enumerate(differences) if v])
        return hash

    def process(self, image: np.ndarray, min_distance: int = 10) -> np.ndarray:
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
        
        min_distance: int, optional
            The min_distance specifies the minimum Hamming distance needed for an image to be classified as 
            unique. The standart value of 10 is based on an article by Dr. Neal Krawetz:
            https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

        Returns
        --------

        np.ndarray
            A copy of the original image array where the duplicate images were removed.
        """

        if not image.ndim == 3:
            raise ValueError

        n, _, _ = image.shape

        hashes = np.zeros(n)
        mask = np.ones(n, dtype=bool)

        cleaned_images = image.copy()


        for i, img in enumerate(image):
            hashes[i] = self._difference_hash(img)
        

        for i in range(n):
            
            idx = i+1

            if idx >= n:
                break
            
            h = hashes[i]

            diff = abs(h - hashes[idx:])
            mask[idx:] &= diff > min_distance

        cleaned_images = cleaned_images[mask]
        
        return cleaned_images
