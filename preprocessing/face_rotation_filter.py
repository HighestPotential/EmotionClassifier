from abc import ABC, abstractmethod
import cv2
import numpy as np
import mediapipe as mp
from image_processor_interface import ImageProcessor

class FaceRotationFilter(ImageProcessor):
    def __init__(self, max_yaw_degrees: float = 25.0):
        self.max_yaw = max_yaw_degrees
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
       
    def process(self, image: np.ndarray) -> bool:
        """
        Calculates face Yaw on a 192x192 resized version of the image.
        Returns True if face is rotated > max_yaw (should be filtered).
        Returns False if face is good.
        """
        if image is None:
            print("Error: Input image is None.")
            return True

        # CHANGED: Resize the image to 192x192 before processing
        image = cv2.resize(image, (192, 192))

        # Working on a copy to avoid modifying the original dataset image for visualization
        debug_image = image.copy()
        img_h, img_w, _ = debug_image.shape # This will now be 192, 192
        face_3d = []
        face_2d = []

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        # If no face is detected, we typically filter it out (return True)
        if not results.multi_face_landmarks:
            print("No face detected.")
            # CHANGED: Show the empty resized image even if no face found, for debugging
            cv2.imshow("Face Rotation Filter", debug_image) 
            cv2.waitKey(100)
            return True

        for face_landmarks in results.multi_face_landmarks:
            # Draw the MediaPipe Mesh (Mask)
            self.mp_drawing.draw_landmarks(
                image=debug_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Indices: Nose(1), Chin(199), Left Eye(33), Right Eye(263), Mouth Left(61), Mouth Right(291)
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 199, 33, 263, 61, 291]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z]) 

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix approximation (based on 192 width)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                singular = sy < 1e-6

                # Calculate Euler angles
                if not singular:
                    x = np.arctan2(rmat[2, 1], rmat[2, 2])
                    y = np.arctan2(-rmat[2, 0], sy) # Yaw
                    z = np.arctan2(rmat[1, 0], rmat[0, 0])
                else:
                    x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                    y = np.arctan2(-rmat[2, 0], sy)
                    z = 0

                yaw_deg = np.degrees(y)
                
                # Check threshold
                should_filter = abs(yaw_deg) > self.max_yaw

                # --- Visualization ---
                color = (0, 0, 255) if should_filter else (0, 255, 0)
                status_text = "REJECT" if should_filter else "ACCEPT"
                
                # CHANGED: Adjusted font scale slightly for smaller resolution
                info = f"Yaw: {int(yaw_deg)}|{status_text}"
                cv2.putText(debug_image, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 1) # Smaller font for 192px

                # Draw nose direction
                nose_2d = (int(face_2d[0][0]), int(face_2d[0][1]))
                p_proj = (int(nose_2d[0] + y * 200), int(nose_2d[1])) # Projection
                cv2.arrowedLine(debug_image, nose_2d, p_proj, color, 2)
                
                cv2.imshow("Face Rotation Filter", debug_image)
                cv2.waitKey(0) # Wait indefinitely
                
                return should_filter

        return True 

if __name__ == "__main__":
    filter_tool = FaceRotationFilter(max_yaw_degrees=30.0)
    
    img_path = r"D:\3thSemester\DLCVProject\EmotionClassifier\datasets\FERPlus\train\disgust\augmented_3.png"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image at {img_path}")
    else:
        is_bad_image = filter_tool.process(img)
        print(f"Should filter image? {is_bad_image}")
    
    cv2.destroyAllWindows()
