import cv2
import numpy as np
import glob
import os
import torch
from tqdm import tqdm
from batch_face import RetinaFace

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# "CHANGED"  -> Show [Original] next to [Fixed]
# "ACCEPTED" -> Show the final upright version
FILTER_MODE = "CHANGED" 
IMAGE_FOLDER = r"D:\3thSemester\DLCVProject\EmotionClassifier\datasets\FER2013\train\sad"

# Minimum confidence to consider it a valid face at all
MIN_CONFIDENCE_THRESHOLD = 0.2

# ---------------------------------------------------------
# HELPER: GRID GENERATOR
# ---------------------------------------------------------
def create_image_grid(images, max_cols=8, thumb_size=(100, 100)):
    if not images: return None
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if w > h * 1.5: 
            target_w, target_h = thumb_size[0] * 2, thumb_size[1]
            resized.append(cv2.resize(img, (target_w, target_h)))
            actual_cols = max(1, max_cols // 2)
        else:
            resized.append(cv2.resize(img, thumb_size))
            actual_cols = max_cols
    n = len(resized)
    cols = min(n, actual_cols)
    rows = int(np.ceil(n / cols))
    cell_h, cell_w = resized[0].shape[:2]
    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for idx, img in enumerate(resized):
        r = idx // cols
        c_idx = idx % cols
        grid[r*cell_h:(r+1)*cell_h, c_idx*cell_w:(c_idx+1)*cell_w] = img
    return grid

# ---------------------------------------------------------
# LOGIC CLASS (Brute Force Check)
# ---------------------------------------------------------
class BruteForceOrientationFixer:
    def __init__(self):
        print("Loading RetinaFace on GPU...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        
        # Use low threshold to catch faces in bad orientations
        self.detector = RetinaFace(gpu_id=0 if self.device == 'cuda' else -1)
        print("Model Loaded.")

    def get_max_face_score(self, image_bgr):
        """Runs detection and returns the score of the largest face found."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Suppress prints from batch_face if any (usually it is silent)
        faces = self.detector(img_rgb, threshold=0.1)
        
        if not faces:
            return 0.0
            
        best_score = 0.0
        max_area = 0
        for box, landmarks, score in faces:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_score = score
        return best_score

    def process(self, image: np.ndarray):
        if image is None: return {'status': "REJECTED", 'original': None}

        rotations = [
            ("Original (0)", None),
            ("Rotated 90 CW", cv2.ROTATE_90_CLOCKWISE),
            ("Rotated 180", cv2.ROTATE_180),
            ("Rotated 90 CCW", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        best_score = -1.0
        best_image = image
        best_rotation_name = "Original (0)"
        
        # -------------------------------------------------
        # BRUTE FORCE LOOP
        # -------------------------------------------------
        for rotation_name, rotate_code in rotations:
            if rotate_code is not None:
                rotated_img = cv2.rotate(image, rotate_code)
            else:
                rotated_img = image
            
            score = self.get_max_face_score(rotated_img)
            
            # REMOVED: print(f" | ".join(scores_log)) 
            
            if score > best_score:
                best_score = score
                best_image = rotated_img
                best_rotation_name = rotation_name

        # -------------------------------------------------
        # DECISION
        # -------------------------------------------------
        if best_score < MIN_CONFIDENCE_THRESHOLD:
             return {'status': "REJECTED", 'original': image}

        was_changed = (best_rotation_name != "Original (0)")
        
        # --- VISUALIZATION ---
        vis_original = image.copy()
        # ÄNDERUNG: Scale von 1 auf 0.6 reduziert
        cv2.putText(vis_original, "Orig", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        fixed_image = best_image.copy()
        # ÄNDERUNG: Scale von 1 auf 0.6 reduziert, Y-Position leicht angepasst (von 30 auf 20)
        cv2.putText(fixed_image, f"Fix: {best_score:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if was_changed:
            h1, w1 = vis_original.shape[:2]
            h2, w2 = fixed_image.shape[:2]
            
            if h2 != h1:
                scale = h1 / h2
                new_w = int(w2 * scale)
                fixed_resized = cv2.resize(fixed_image, (new_w, h1))
            else:
                fixed_resized = fixed_image

            separator = np.zeros((h1, 5, 3), dtype=np.uint8)
            vis_pair = np.hstack([vis_original, separator, fixed_resized])
            # ÄNDERUNG: Pfeil auch etwas dünner gemacht (von 4 auf 2)
            cv2.arrowedLine(vis_pair, (w1//2, h1//2), (w1 + 5 + w2//2, h1//2), (0, 255, 0), 2)
        else:
            vis_pair = vis_original
            
        return {
            'status': "CHANGED" if was_changed else "ALREADY_GOOD",
            'original': vis_original,
            'fixed': fixed_image,
            'vis_pair': vis_pair
        }

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    fixer = BruteForceOrientationFixer()
    
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    
    print(f"Scanning {len(image_files)} images...")
    print(f"Mode: {FILTER_MODE}")
    
    display_list = []
    
    # REMOVED: explicit print inside loop. 'tqdm' handles the UI now.
    for fpath in tqdm(image_files, desc="Fixing Rotations"):
        img = cv2.imread(fpath)
        if img is None: continue
        
        result = fixer.process(img)
        status = result['status']
        
        if FILTER_MODE == "CHANGED":
            if status == "CHANGED":
                display_list.append(result['vis_pair'])
        elif FILTER_MODE == "ACCEPTED":
            if status == "CHANGED" or status == "ALREADY_GOOD":
                display_list.append(result['fixed'])

    print("-" * 30)
    print(f"Found {len(display_list)} items for display.")
    
    if display_list:
        thumb_w = 200 if FILTER_MODE == "CHANGED" else 100
        thumb_h = 100
        final_grid = create_image_grid(display_list, max_cols=6, thumb_size=(thumb_w, thumb_h))
        cv2.imshow(f"Visualizer: {FILTER_MODE}", final_grid)
        print("Press any key to close window...")
        cv2.waitKey(0)
    else:
        print("No images matched the filter.")
        
    cv2.destroyAllWindows()
