from concurrent.futures import ProcessPoolExecutor

class Pipeline:
    def __init__(self, steps: list[IImageProcessor]):
        self.steps = steps

    def execute(self, img):
        for step in self.steps:
            img = step.process(img) # Guarantees this method exists
        return img
