import cv2
import numpy as np
import math
import glob
import os
from image_processor_interface import ImageProcessor
from abc import ABC, abstractmethod

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from mtcnn import MTCNN
import time

# Options: "ACCEPTED" (show good images) or "REJECTED" (show bad images)
FILTER_MODE = "REJECTED" 

# Path to your dataset
IMAGE_FOLDER = r"D:\3thSemester\DLCVProject\EmotionClassifier\datasets\FER2013\train\sad"


class FaceRotationFilter(ImageProcessor):
    def __init__(self):
        print("Loading MTCNN Model... (Please wait)")
        self.detector = MTCNN()

    def process(self, image: np.ndarray) -> tuple[bool, np.ndarray]:
        if image is None: return False, np.array([])
        
        # MTCNN needs RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        debug_view = image.copy()
        
        # 1. Detect Faces
        try:
            results = self.detector.detect_faces(img_rgb)
        except: # the image format is corrupted, there is a memory error, or the MTCNN model crashes
            return True, debug_view 
        
        # If no face found
        if not results:
            h, w = debug_view.shape[:2]
            cv2.line(debug_view, (0, 0), (w, h), (0, 0, 255), 2)
            cv2.line(debug_view, (0, h), (w, 0), (0, 0, 255), 2)
            cv2.putText(debug_view, "NO FACE", (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return True, debug_view # True = REJECTED

        # 2. Analyze Primary Face
        face = results[0]
        box = face['box']
        keypoints = face['keypoints']
        
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        nose = keypoints['nose']
        
        # 3. Geometric Logic (Symmetry)
        dist_left = math.dist(left_eye, nose)
        dist_right = math.dist(right_eye, nose)
        total_span = dist_left + dist_right
        
        is_rejected = False
        color = (0, 255, 0) # Green
        
        if total_span > 0:
            ratio = dist_left / total_span
            
            # Threshold: 0.35 to 0.65 is "Frontal"
            if ratio < 0.25 or ratio > 0.75:
                is_rejected = True
                color = (0, 0, 255) # Red
                status = f"REJ:{ratio:.2f}"
            else:
                status = f"OK:{ratio:.2f}"

            # Visualization
            cv2.circle(debug_view, left_eye, 3, (255, 0, 0), -1)
            cv2.circle(debug_view, right_eye, 3, (255, 0, 0), -1)
            cv2.circle(debug_view, nose, 3, (0, 255, 255), -1)
            
            x, y, w, h = box
            cv2.rectangle(debug_view, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_view, status, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return is_rejected, debug_view


def create_image_grid(images, max_cols=10, thumb_size=(80, 80)):
    if not images: return None
    resized = [cv2.resize(img, thumb_size) for img in images]
    n = len(resized)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    w, h = thumb_size
    c = resized[0].shape[2] 
    
    grid = np.zeros((h * rows, w * cols, c), dtype=np.uint8)
    for idx, img in enumerate(resized):
        r = idx // cols
        c_idx = idx % cols
        grid[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w] = img
    return grid


if __name__ == "__main__":
    processor = FaceRotationFilter()
    
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    
    print(f"Checking {len(image_files)} images for mode: {FILTER_MODE}...")
    
    results_list = []

    for fpath in image_files:
        img = cv2.imread(fpath)
        if img is None: continue


        img_upscaled = cv2.resize(img, (300, 300))
            
        is_rejected, result_view = processor.process(img_upscaled)
        
        # LOGIC: Filter based on the string variable
        if FILTER_MODE == "REJECTED":
            if is_rejected:
                results_list.append(result_view)
        elif FILTER_MODE == "ACCEPTED":
            if not is_rejected:
                results_list.append(result_view)
    
    print("-" * 30)
    print(f"Total {FILTER_MODE}: {len(results_list)}")
    
    if results_list:
        print("Displaying grid...")
        final_grid = create_image_grid(results_list, max_cols=12)
        cv2.imshow(f"Result: {FILTER_MODE}", final_grid)
        cv2.waitKey(0)
    else:
        print(f"No images found for mode: {FILTER_MODE}")
    
    cv2.destroyAllWindows()
