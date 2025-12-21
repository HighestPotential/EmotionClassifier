import cv2
import numpy as np
import sys
from image_processor_interface import ImageProcessor

# Robust Import for MediaPipe
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError:
    print("CRITICAL: MediaPipe not found. Ensure you are in the 'emotion_cls' environment.")
    sys.exit(1)
except AttributeError:
    print("CRITICAL: MediaPipe 'solutions' not found. Check for namespace shadowing.")
    sys.exit(1)

class FaceRotationFilter(ImageProcessor):
    def __init__(self):
        # Initialize resources once to avoid overhead
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_head_pose(self, image, face_landmarks):
        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []

        # Landmarks: Nose(1), Chin(199), Left Eye(33), Right Eye(263), Mouth L(61), Mouth R(291)
        lm_indices = [1, 199, 33, 263, 61, 291]

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in lm_indices:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera Matrix (approximate)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        # Get Angles
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        return angles[0] * 360, angles[1] * 360, angles[2] * 360

    def process(self, img) -> bool:
        if img is None:
            return False

        # 1. Resize Image
        img_resized = cv2.resize(img, (192, 192)) # CHANGED: Resizing to 192x192 as requested
        
        # 2. Process with MediaPipe
        results = self.face_mesh.process(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw Mesh
                mp_drawing.draw_landmarks(
                    image=img_resized,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Calculate Pose
                pitch, yaw, roll = self.get_head_pose(img_resized, face_landmarks)
                
                # Display Info (CORRECTED SYNTAX HERE)
                cv2.putText(img_resized, f"P: {int(pitch)}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img_resized, f"Y: {int(yaw)}",   (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img_resized, f"R: {int(roll)}",  (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display window
        cv2.imshow('Face Rotation Filter (192x192)', img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True

# --- Testing Block ---
if __name__ == "__main__":
    image_path = r'datasets\FER2013\train\fear\Training_531060.jpg'
    
    # Mocking the interface for standalone testing if the file is missing
    if 'ImageProcessor' not in globals():
        class ImageProcessor: pass

    processor = FaceRotationFilter()
    original_img = cv2.imread(image_path)
    
    if original_img is None:
        print(f"Error: Could not load {image_path}")
    else:
        processor.process(original_img)
