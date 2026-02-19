from concurrent.futures import ProcessPoolExecutor
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class Pipeline:
    def __init__(self, steps: list[ImageProcessor]):
        self.steps = steps

    def execute(self, img):
        for step in self.steps:
            try:
                img = step.process(img)
            except SkipImage as e: 
                return None
        return img
