import numpy as np
<<<<<<< HEAD
import cv2
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class RemoveBlurredFaces(ImageProcessor):
    """
    Filters images using Fast Fourier Transform (FFT) to detect blur.
    This is more robust than Laplacian variance for smooth faces.
    """

    def process(self, image: np.ndarray, threshold: float = 10.0) -> np.ndarray:
        """
        :param threshold: FFT magnitude threshold. 
                          - Below 10 is usually very blurry.
                          - 10-20 is soft.
                          - Above 20 is sharp.
        """
        if image is None:
            raise SkipImage("Image is None")

        # 1. Convert to Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))

        # 2. Compute FFT (Fast Fourier Transform) to get frequency domain
        fft = np.fft.fft2(gray)
        fftShift = np.fft.fftshift(fft)

        # 3. Remove low frequencies (the center of the spectrum)
        # We zero out the center 60x60 pixels to inspect only high frequencies (edges/details)
        fftShift[cY - 30:cY + 30, cX - 30:cX + 30] = 0
        
        # 4. Reconstruct the image with only high frequencies
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        
        # 5. Calculate magnitude of the reconstruction
        magnitude = 20 * np.log(np.abs(recon))
        mean_magnitude = np.mean(magnitude)

        # Check against threshold
        if mean_magnitude < threshold:
             raise SkipImage(f"Image is blurry (FFT score={mean_magnitude:.2f} < {threshold})")

        return image
=======
import cv2 as cv

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

    def process(self, image: np.ndarray, labels: np.ndarray, threshold = 100.0) -> np.ndarray:
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
        
        if not image.ndim == 3:
            raise ValueError

        cleaned_images = image.copy()
        cleaned_labels = labels.copy()

        variances = [cv.Laplacian(sample, cv.CV_64F).var() for sample in cleaned_images]
        variances = np.array(variances)

        mask = variances >= threshold
        cleaned_images = cleaned_images[mask]
        cleaned_labels = cleaned_labels[mask]

        return cleaned_images, cleaned_labels
>>>>>>> main
