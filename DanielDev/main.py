import os
from image_duplicates_v2 import RemoveDuplicates

def load_images():
    paths = []
    images =  os.listdir(os.path.expanduser("./data"))
    for img in images:
        imgPath = os.path.join(os.getcwd(), "data", img)
        paths.append(imgPath)
    
    return paths

if __name__ == "__main__":
    handler = RemoveDuplicates()

    images = load_images()
    clear = handler.clean_data(images)
    print(len(images), len(clear))