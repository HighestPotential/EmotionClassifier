from concurrent.futures import ProcessPoolExecutor
from image_processor_interface import ImageProcessor

class Pipeline:
    def __init__(self, steps: list[ImageProcessor]):
        self.steps = steps

    def execute(self, img):
        for step in self.steps:
            img = step.process(img) # Guarantees this method exists
        return img
