import os  # CHANGED: Imported os to handle Linux file paths dynamically
import dataset_runner
from pipeline import Pipeline
from image_duplicates import RemoveDuplicates
from blurred_images import RemoveBlurredFaces
from correct_roll import FaceOrientationFilter
from cropp_face import CroppingFace
from lighting_filter import LightingFilter
from resizing_64 import ResizingTo64
from to_rgb import ToRGB
from yaw_rotation_filter import FaceRotationFilter
from face_exists import FaceExistenceFilter

if __name__ == "__main__":
    duplicate_remover = RemoveDuplicates() # TODO: find a solution how to use it
    
    steps = [FaceExistenceFilter(confidence_threshold=0.8)]

#FaceRotationFilter(), FaceOrientationFilter(),\
#        LightingFilter(), RemoveBlurredFaces(),\
#             CroppingFace(), FaceExistenceFilter(), ResizingTo64(), ToRGB()
    pipeline = Pipeline(steps)

    # '~' automatically expands to '/home/dumanskyy'
    base_path = os.path.expanduser("~/work/EmotionClassifier")

    input_path = os.path.join(base_path, "latest_3_0_ready_to_use_datasets/EmoSet-118k")
    output_path = os.path.join(base_path, "EmoSet_strong_face_detection_80")

    print(f"Reading from: {input_path}") 
    
    dataset_runner.run_folder(pipeline, 
        input_path, 
        output_path, 
        keep_structure=True, max_files=None, log_every=1000)
