from concurrent.futures import ProcessPoolExecutor
from image_processor_interface import ImageProcessor
from skip_image import SkipImage

class Pipeline:
    def __init__(self, steps: list[ImageProcessor]):
        self.steps = steps

    def execute(self, img):
        try:
            for step in self.steps:
                img = step.process(img)
            return img
        except SkipImage:
            return None
