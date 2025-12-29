import cv2
import numpy as np
from image_processor_interface import ImageProcessor
from skip_image import SkipImage
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt



class cropping_face(ImageProcessor):
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
      
    def process(self, image: np.ndarray, threshold=0.2) -> np.ndarray:
        """
        - 

        :param image: supports (H,W) and (H,W,C) shape

        :type image: expects a np.ndarray

        :return:        - Returns None when the input "image" is None or empty 
                        - Returns the cropped face when the face area represents less than 20% of the image
                        - Returns the normal image with the marked face 

        :rtype: ndarray
        """

        

        if image is None:
                raise SkipImage("Input image is None")
        


        normal_image = load_image(image)
        normal_image = np.clip(normal_image, 0, 255).astype(np.uint8) #garantues the np.uint8 type between 0 and 255 values
        copy_image = normal_image.copy()

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w


        mtcnn = MTCNN(device="CPU:0")

        # Detect faces and landmarks
        result = mtcnn.detect_faces(image, threshold_onet=0.85)

        if len(result) == 0:
            raise SkipImage("Face not found")
            
        else: 
            
            # Take the first detected face
            x, y, w, h = result[0]["box"]
            face_area = w * h
            face_ratio = face_area / img_area

            if face_ratio < threshold:
                # Crop face
                face_cropped = copy_image[y:y+h, x:x+w]

                return face_cropped
            else:
                return copy_image


            
        