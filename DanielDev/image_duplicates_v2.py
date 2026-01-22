import numpy as np
import cv2 as cv
from dataclasses import dataclass
from PIL import Image

class RemoveDuplicates:

    def _difference_hash(self, image: np.ndarray, hashsize: int = 8) -> int:
        """
        Removes items from a list of images according to the class description

        Parameters
        -----------

        image: np.ndarray
            A numpy array that stores images. The shape of the array is expected to be (n, h, w)
            where:
                - n: number of images
                - h: height of each image
                - w: width of each image
        
        hashsize: int, optional
            The size the image will be shrunken to to compute the hash. With a 
            hashsize of n the resulting hash will be of length 8x8 Bits

        Returns
        --------

        int
            The computed distance hash
        """
        resized: np.ndarray = cv.resize(image, (hashsize + 1, hashsize))
        
        differences: np.ndarray = resized[:, 1:] > resized[:, :-1]
        differences = differences.flatten()

        hash = sum([2**i for (i, v) in enumerate(differences) if v])
        return hash

    def _load_images(self, strBatch: list[str]) -> np.ndarray:
        images = []

        for path in strBatch:
            img = Image.open(path)
            img_arr = np.array(img).astype(np.uint8)
            images.append(img_arr)
        
        return np.array(images)

    def clean_data(self, images: list[str], min_distance: int = 10, batch_size: int = 128) -> None:
        bucketMap: dict[int:list[tuple[int, str]]] = {}
        
        numImages = len(images)
        numPasses = (numImages // batch_size) + 1
        
        for i in range(numPasses):
            strBatch = images[i*batch_size: (i+1)*batch_size]
            imgBatch = self._load_images(strBatch=strBatch)
            differences = list(map(self._difference_hash, imgBatch))

            entries = list(zip(differences, strBatch))
            
            for entry in entries:
                diff, _ = entry
                f = lambda x: abs(diff - x[0]) > min_distance
                strDiff = str(diff)

                index = int(strDiff[:10]) if len(strDiff) > 10 else 0
                similar = bucketMap.setdefault(index, [])

                distancesOk = list(map(f, similar))
                if all(distancesOk):
                    similar.append(entry)
        
        return bucketMap

