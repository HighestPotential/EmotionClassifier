from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch
from sixdrepnet import SixDRepNet
from batch_face import RetinaFace
from image_processor_interface import ImageProcessor

class FaceRotationFilter(ImageProcessor):
    def __init__(self, yaw_threshold_degrees: float = 45.0):
        print("Loading Models...")
        
        # 1. Check GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ WARNING: GPU not found. Running on CPU.")

        # 2. Initialize RetinaFace (PyTorch)
        # gpu_id=0 tells it to use the GPU. 
        self.detector = RetinaFace(gpu_id=0 if self.device == 'cuda' else -1)
        
        # 3. Initialize 6DRepNet (PyTorch)
        self.pose_model = SixDRepNet(gpu_id=0 if self.device == 'cuda' else -1)
        
        self.yaw_threshold = yaw_threshold_degrees
        print(f"Models Loaded. Yaw Threshold: {self.yaw_threshold} degrees")

def process(self, image: np.ndarray) -> np.ndarray:
        """
        Evaluates an image using **RetinaFace (PyTorch)** for detection and **6DRepNet** for pose estimation
        to determine if the face is front-facing.

        The pipeline executes the following steps:
        1.  **Preprocessing:** Checks if the input is Grayscale (2D) and converts it to BGR (3D) if necessary.
            Then converts BGR to RGB, as required by the models.
        2.  **Detection:** Uses **RetinaFace** (threshold=0.5) to locate the primary face.
        3.  **Selection:** Crops the largest detected face from the original image.
        4.  **Analysis:** Feeds the crop into **6DRepNet** to predict the 3D rotation angles (Pitch, Yaw, Roll).
        5.  **Filtering:** Returns the image only if the absolute Yaw angle is within `self.yaw_threshold`.

        Args:
            image (np.ndarray): The input image array. It should be in **BGR** format (standard OpenCV).
                                - Supports 3-channel images (H, W, 3).
                                - Supports 1-channel Grayscale images (H, W) or (H, W, 1), which will 
                                  be converted to 3-channel BGR internally.

        Returns:
            np.ndarray | None: Returns the original `image` reference if it passes all checks.
            Returns `None` if:
                - The input `image` is None or empty.
                - No face is detected by RetinaFace.
                - The 6DRepNet prediction fails or the Yaw angle > `yaw_threshold`.
        """
        if image is None:
            return None
        
        # RetinaFace expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        
        # 1. Detect Faces
        # Using threshold 0.5 to ensure we catch rotated faces
        faces = self.detector(img_rgb, threshold=0.5)

        if not faces:
            return None # REJECTED: No face found

        # 2. Analyze Primary Face (Take the first/largest one)
        # batch_face returns a list of tuples: (box, landmarks, score)
        box, landmarks, score = faces[0]
        x1, y1, x2, y2 = map(int, box)
        
        # Safety Clamping to avoid crash on edge crops
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None # REJECTED: Invalid crop

        # 3. 6DRepNet Prediction
        try:
            _pitch, yaw, _roll = self.pose_model.predict(face_crop)
            
            # Filter based on Yaw (Left/Right rotation)
            if abs(yaw) > self.yaw_threshold:
                return None # REJECTED: Rotation too extreme
            
            # ACCEPTED: Return the original clean image
            return image
            
        except Exception:
            return None # REJECTED: Prediction error
