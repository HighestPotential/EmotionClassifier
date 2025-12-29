import cv2
import numpy as np
import glob
import os
import torch
from tqdm import tqdm
from sixdrepnet import SixDRepNet

from batch_face import RetinaFace

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
#REJECTED or ACCEPTED
FILTER_MODE = "ACCEPTED"
YAW_THRESHOLD_DEGREES = 45.0
IMAGE_FOLDER = r"D:\3thSemester\DLCVProject\EmotionClassifier\datasets\FER2013\train\sad"

class FaceRotationFilter:
    def __init__(self):
        print("Loading Models on GPU...")
        
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
        # Passing gpu_id=0 ensures it puts weights on the GPU
        self.pose_model = SixDRepNet(gpu_id=0 if self.device == 'cuda' else -1)
        
        print("Models Loaded Successfully.")

    def process(self, image: np.ndarray) -> tuple[bool, np.ndarray]:
        if image is None: return False, np.array([])
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        debug_view = image.copy()
        h_orig, w_orig = image.shape[:2]
        
        # -------------------------------------------------
        # STEP 1: Detect Faces (PyTorch Engine)
        # -------------------------------------------------
        # batch-face returns a list of faces directly.
        # No need for complex dictionary parsing.
        faces = self.detector(img_rgb, threshold=0.60)

        if not faces:
            cv2.line(debug_view, (0, 0), (w_orig, h_orig), (0, 0, 255), 2)
            cv2.line(debug_view, (0, h_orig), (w_orig, 0), (0, 0, 255), 2)
            return True, debug_view # True = REJECTED

        # 2. Analyze Primary Face (Take the largest/first one)
        # batch_face returns: [[x1, y1, x2, y2, score], landmarks...]
        # We take the first box.
        box, landmarks, score = faces[0]
        x1, y1, x2, y2 = map(int, box)
        
        # Safety Clamping
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        face_crop = image[y1:y2, x1:x2]
        
        is_rejected = False
        color = (0, 255, 0)
        status_text = "OK"

        if face_crop.size > 0:
            # 3. 6DRepNet Prediction (Already on GPU)
            try:
                pitch, yaw, roll = self.pose_model.predict(face_crop)
                
                if abs(yaw) > YAW_THRESHOLD_DEGREES:
                    is_rejected = True
                    color = (0, 0, 255)
                    status_text = f"REJ:{int(yaw)}"
                else:
                    status_text = f"OK:{int(yaw)}"
            except Exception:
                status_text = "ERR"
                is_rejected = True
        else:
            is_rejected = True
            status_text = "BadCrop"

        cv2.rectangle(debug_view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug_view, status_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
        return is_rejected, debug_view

# Helper function for grid (Same as before)
def create_image_grid(images, max_cols=10, thumb_size=(80, 80)):
    # CHANGED: removed labels parameter and removed reserved label_area so no black line appears under thumbnails
    if not images: return None
    resized = [cv2.resize(img, thumb_size) for img in images]
    n = len(resized)
    cols = min(n, max_cols)
    rows = int(np.ceil(n / cols))
    w, h = thumb_size
    shape = resized[0].shape
    c = shape[2] if len(shape) > 2 else 1
    grid = np.zeros((h * rows, w * cols, c), dtype=np.uint8)  # CHANGED: no extra vertical space
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
    
    results_list = []  # CHANGED: removed labels_list as grid no longer needs separate label area

    for fpath in tqdm(image_files, desc="Processing"):
        img = cv2.imread(fpath)
        if img is None: continue
        
        # Upscaling is still good for 64x64 images
        img_upscaled = cv2.resize(img, (300, 300))
        is_rejected, result_view = processor.process(img_upscaled)
        
        # CHANGED: derive a label for the image (use filename without extension)
        basename = os.path.basename(fpath)
        label_to_show = os.path.splitext(basename)[0]  # CHANGED: filename without extension

        # Overlay the filename on the bottom of the result_view with large red letters and a black outline
        h_r, w_r = result_view.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.4                    # CHANGED: make filename text much larger
        thickness = 5                  # CHANGED: thick text for visibility
        red_color = (0, 0, 255)        # CHANGED: red text (BGR)
        outline_color = (0, 0, 0)      # CHANGED: black outline color for readability

        (text_w, text_h), baseline = cv2.getTextSize(label_to_show, font, scale, thickness)
        x_text = max(10, (w_r - text_w) // 2)
        y_text = h_r - 15

        # CHANGED: remove filled background rectangle entirely (this removed the big black bar)
        # Instead draw an outline by drawing the text twice: thick black (outline) then red on top
        cv2.putText(result_view, label_to_show, (x_text, y_text), font, scale, outline_color, thickness + 2, cv2.LINE_AA)  # CHANGED: black outline
        cv2.putText(result_view, label_to_show, (x_text, y_text), font, scale, red_color, thickness, cv2.LINE_AA)          # CHANGED: red foreground

        if FILTER_MODE == "REJECTED" and is_rejected:
            results_list.append(result_view)
        elif FILTER_MODE == "ACCEPTED" and not is_rejected:
            results_list.append(result_view)
    
    print("-" * 30)
    print(f"Total {FILTER_MODE}: {len(results_list)}")
    
    if results_list:
        final_grid = create_image_grid(results_list, max_cols=12)  # CHANGED: grid no longer uses labels param
        cv2.imshow(f"Result: {FILTER_MODE}", final_grid)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
