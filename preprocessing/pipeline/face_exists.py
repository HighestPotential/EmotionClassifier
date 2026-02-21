import cv2
import numpy as np
import torch
from batch_face import RetinaFace 
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class FaceExistenceFilter(ImageProcessor):
    """
    A class that simply checks if a face exists in the image using RetinaFace.
    If a face is found, it returns the original image.
    If no face is found, it raises SkipImage.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        
        # Initialize detector
        self.detector = RetinaFace(gpu_id=0 if self.device == 'cuda' else -1)
        print(f"FaceExistenceFilter initialized on {self.device} (threshold={self.confidence_threshold})")

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: Input image (H, W, C)
        :return: Original image if face detected, raises SkipImage otherwise.
        """
        if image is None:
            raise SkipImage("Input image is None")
            
        if image.size == 0:
            raise SkipImage("Input image is Empty")
       
        # Prepare image for detection (RGB)
        # We work on a lightweight copy for detection to avoid altering the original
        img_forcv = image
        if len(img_forcv.shape) == 2:
            img_forcv = cv2.cvtColor(img_forcv, cv2.COLOR_GRAY2BGR)
            
        img_rgb = cv2.cvtColor(img_forcv, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.detector(img_rgb, threshold=self.confidence_threshold)

        if not faces:
            # CHANGED: Raise SkipImage if the list of faces is empty
            raise SkipImage("Filter dropped image: RetinaFace failed to detect any face at the end")
        
        # CHANGED: Return the original unmodified image if a face exists
        return image
