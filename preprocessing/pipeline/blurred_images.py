import numpy as np
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
