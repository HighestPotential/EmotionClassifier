from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch
from batch_face import RetinaFace
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class FaceOrientationFilter(ImageProcessor):
    def __init__(self, min_confidence: float = 0.4):
        """
        Initializes the RetinaFace detector on GPU (if available).
        
        Args:
            min_confidence (float): The minimum score required for the 'best' rotation 
                                    to be accepted as a valid face. Defaults to 0.5.
        """
        self.min_confidence = min_confidence
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize RetinaFace
        # We use a specific gpu_id if CUDA is available, otherwise -1 for CPU
        self.detector = RetinaFace(gpu_id=0 if self.device == 'cuda' else -1)

    def _get_max_face_score(self, image_bgr: np.ndarray) -> float:
        """
        Internal helper: Returns the confidence score of the single largest face in the image.
        Returns 0.0 if no faces are found.
        """
        # RetinaFace expects RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Use a very low threshold (0.05) here because we are just comparing scores
        # between rotations. We filter by self.min_confidence later.
        faces = self.detector(img_rgb, threshold=0.05)
        
        if not faces:
            return 0.0
            
        best_score = 0.0
        max_area = 0
        
        # Find the largest face by area and take its score
        for box, landmarks, score in faces:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_score = score
                
        return best_score

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Determines the correct orientation of the image by testing 4 rotations (0, 90, 180, 270)
        and selecting the one where the face detector is most confident.

        Args:
            image (np.ndarray): Input image in BGR (3-channel) or Grayscale (2-channel) format.

        Returns:
            np.ndarray | None: 
                - Returns the image rotated to the upright position if a valid face is found.
                - Returns None if no face is detected in any orientation (or score < min_confidence).
        """
        if image is None:
            raise SkipImage("Input image is None")

        # 1. Handle Grayscale -> BGR conversion
        # RetinaFace requires 3 channels.
        if len(image.shape) == 2:  # (H, W)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1: # (H, W, 1)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        # 2. Define rotations to test
        rotations = [
            (None, image_bgr), # 0 degrees
            (cv2.ROTATE_90_CLOCKWISE, None),
            (cv2.ROTATE_180, None),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, None),
        ]

        best_score = -1.0
        best_image = None

        # 3. Brute-force check all 4 orientations
        for rotate_code, cached_img in rotations:
            # Perform rotation if needed
            if rotate_code is not None:
                current_img = cv2.rotate(image_bgr, rotate_code)
            else:
                current_img = cached_img

            # Get score for this orientation
            score = self._get_max_face_score(current_img)

            if score > best_score:
                best_score = score
                best_image = current_img

        # 4. Final Verification
        # If even the best rotation has a low score, it's likely not a face at all.
        if best_score < self.min_confidence:
            return SkipImage(f"No face detected above confidence {self.min_confidence}")

        return best_image
