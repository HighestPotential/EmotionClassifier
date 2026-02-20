import cv2
import numpy as np
import torch
from batch_face import RetinaFace 
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class CroppingFace(ImageProcessor):
    """
    A class that detects and crops the human face using RetinaFace.
    """
    def __init__(self, confidence_threshold: float = 0.5, threshold_area: float = 0.4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.threshold_area = threshold_area
        
        # Initialize detector
        self.detector = RetinaFace(gpu_id=0 if self.device == 'cuda' else -1)
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        :param threshold: The ratio (face_area / img_area) below which cropping occurs.
        """
        if image is None:
            raise SkipImage("Input image is None")
            
        if image.size == 0:
            raise SkipImage("Input image is Empty")
       
        # Work on a copy
        image_copy = image.copy()
        
        # Ensure 3 channels
        if len(image_copy.shape) == 2:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
            
        img_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # Detect faces
        faces = self.detector(img_rgb, threshold=self.confidence_threshold)

        if not faces:
            # If no face is found, we cannot crop. Raise error to filter this image out.
            raise SkipImage("RetinaFace failed to detect any face")
            
        else: 
            best_face = None
            max_face_area = 0

            # Find the largest face
            for box, landmarks, score in faces:
                x1, y1, x2, y2 = map(int, box)
                area_temp = (x2 - x1) * (y2 - y1)
                
                if area_temp > max_face_area:
                    max_face_area = area_temp
                    best_face = box

            x1, y1, x2, y2 = map(int, best_face)
            
            w_raw = x2 - x1
            h_raw = y2 - y1

            face_ratio = max_face_area / img_area

            # Padding logic
            padding_ratio_w = 0.4
            padding_ratio_h = 0.2  
            pad_w = int(w_raw * padding_ratio_w) 
            pad_h = int(h_raw * padding_ratio_h) 

            # Apply padding to coordinates
            new_x1 = max(0, x1 - pad_w)
            new_y1 = max(0, y1 - pad_h)
            new_x2 = min(img_w, x2 + pad_w)
            new_y2 = min(img_h, y2 + pad_h)
            
            new_w = new_x2 - new_x1
            new_h = new_y2 - new_y1
            
            # Check if the face is small enough to warrant cropping
            if face_ratio < self.threshold_area:
                if new_w <= 0 or new_h <= 0:
                    raise SkipImage("Calculated crop dimensions are invalid")
                
                # Perform the crop
                face_cropped = image_copy[new_y1:new_y2, new_x1:new_x2]

                # CHANGED: Removed debug saving and print statements for production use
                return face_cropped
            else:
                # If face is already large enough, return original
                return image_copy
