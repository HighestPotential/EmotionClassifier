from abc import ABC, abstractmethod
import cv2
import numpy as np
import mediapipe as mp
import glob
import os
import math
from image_processor_interface import ImageProcessor

class FaceRotationFilter(ImageProcessor):
    def __init__(self, max_yaw_degrees: float = 25.0):
        self.max_yaw = max_yaw_degrees
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        
        # CHANGED: Defined a Generic 3D Face Model (Standard for PnP)
        # Points: Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
        self.generic_face_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left Eye Left Corner
            (225.0, 170.0, -135.0),      # Right Eye Right Corner
            (-150.0, -150.0, -125.0),    # Left Mouth Corner
            (150.0, -150.0, -125.0)      # Right Mouth Corner
        ], dtype=np.float64)

    def process(self, image: np.ndarray) -> tuple[bool, np.ndarray | None]:
        if image is None: return True, None

        image = cv2.resize(image, (192, 192))
        debug_image = image.copy()
        img_h, img_w, _ = debug_image.shape 

        rgb_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            cv2.putText(debug_image, "NO FACE", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return True, debug_image

        for face_landmarks in results.multi_face_landmarks:
            face_2d = []

            # CHANGED: Use specific landmarks that match our Generic 3D Model
            # MediaPipe Indices: Nose(1), Chin(199), Left Eye(33), Right Eye(263), Mouth Left(61), Mouth Right(291)
            for idx in [1, 199, 33, 263, 61, 291]:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])

            face_2d = np.array(face_2d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # CHANGED: Solve PnP using self.generic_face_3d instead of dynamic 3d points
            success, rot_vec, trans_vec = cv2.solvePnP(self.generic_face_3d, face_2d, cam_matrix, dist_matrix)

            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                singular = sy < 1e-6

                if not singular:
                    x = np.arctan2(rmat[2, 1], rmat[2, 2])
                    y = np.arctan2(-rmat[2, 0], sy) # Yaw
                    z = np.arctan2(rmat[1, 0], rmat[0, 0])
                else:
                    x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                    y = np.arctan2(-rmat[2, 0], sy)
                    z = 0

                yaw_deg = np.degrees(y)
                
                # CHANGED: Debug print to verify angles are actually being calculated
                # print(f"Calculated Yaw: {yaw_deg:.2f}") 

                should_filter = abs(yaw_deg) > self.max_yaw

                if should_filter:
                    color = (0, 0, 255)
                    info = f"Yaw:{int(yaw_deg)}"
                    
                    self.mp_drawing.draw_landmarks(
                        image=debug_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    cv2.putText(debug_image, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, color, 2)

                    nose_2d = (int(face_2d[0][0]), int(face_2d[0][1]))
                    
                    # Project nose direction
                    p_proj = (int(nose_2d[0] + y * 200), int(nose_2d[1]))
                    cv2.arrowedLine(debug_image, nose_2d, p_proj, color, 2)
                    
                    return True, debug_image
                
                return False, None

        return True, debug_image

def create_image_grid(images, max_cols=8):
    if not images: return None
    num_images = len(images)
    h, w, c = images[0].shape
    cols = min(num_images, max_cols)
    rows = math.ceil(num_images / cols)
    grid_h = rows * h
    grid_w = cols * w
    grid_image = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid_image[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return grid_image

if __name__ == "__main__":
    folder_path = r"D:\3thSemester\DLCVProject\EmotionClassifier\datasets\AffectNet\Test\Anger"
    file_patterns = ["*.jpg", "*.png"]
    
    # 30 degrees is quite high. Faces must be turned significantly to trigger it.
    filter_tool = FaceRotationFilter(max_yaw_degrees=30.0)
    
    image_files = []
    for pattern in file_patterns:
        search_query = os.path.join(folder_path, pattern)
        image_files.extend(glob.glob(search_query))
    
    print(f"Found {len(image_files)} images. Processing...")

    rejected_images_list = []

    for file_path in image_files:
        img = cv2.imread(file_path)
        if img is None: continue
            
        is_bad, debug_img = filter_tool.process(img)
        
        if is_bad and debug_img is not None:
            rejected_images_list.append(debug_img)

    print(f"Done. Total rejected: {len(rejected_images_list)} / {len(image_files)}")

    if rejected_images_list:
        final_grid = create_image_grid(rejected_images_list, max_cols=8)
        cv2.imshow("Rejected Images", final_grid)
        cv2.waitKey(0)
    else:
        print("No images were rejected!")
    
    cv2.destroyAllWindows()
