import cv2
import numpy as np
from image_processor_interface import ImageProcessor
from skip_image import SkipImage
from mtcnn import MTCNN
from skip_image import SkipImage


class CroppingFace(ImageProcessor):
    """
    A class that marks the human face with the eyes, mouth and nose in an image using MTCNN.


    Attributes
    -----------

    None

    Methods
    --------

    process(self, image: np.ndarray, threshold = 100.0) -> np.ndarray
        processes n images stored in a numpy array of size (n, h, w) where h and 
        w are the height and width of the image
    """
    def __init__(self):
        self.mtcnn = MTCNN(device="CPU:0")
        
    def process(self, image: np.ndarray, threshold=0.2) -> np.ndarray:
        """
        :param image: supports (H,W) and (H,W,C) shape

        :type image: expects a np.ndarray

        :return:        - Returns None when the input "image" is None or empty 
                        - Returns the cropped face when the face area represents less than 20% of the image
                        - Returns the normal image with the marked face 

        :rtype: ndarray | SkipImage
        """

        

        if image is None:
                raise SkipImage("Input image is None")
            
        if image.size == 0:
            raise SkipImage("Input image is Empty")
       
        image_copy = image.copy()
        image_copy= cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        normal_image = image_copy
        normal_image = np.clip(normal_image, 0, 255).astype(np.uint8) #garantues the np.uint8 type between 0 and 255 values
        copy_image = normal_image.copy()

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w


        

        # Detect faces and landmarks
        result = self.mtcnn.detect_faces(normal_image)

        if len(result) == 0:
            return copy_image
            
        else: 
            
            # Take the first detected face
            x, y, w, h = result[0]["box"]
            face_area = w * h
            face_ratio = face_area / img_area

            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            
            if face_ratio < threshold:
                
                if w <= 0 or h <= 0:
                    # If detection is invalid, return original image instead of crashing
                    return copy_image
                
                # Crop face
                face_cropped = copy_image[y:y+h, x:x+w]

                return face_cropped
            else:
                return copy_image
