import math
import numpy as np
import cv2 as cv
from PIL import Image

class FindDuplicates:

    def __init__(self, hash_function):
        if hash_function == "dHash":
            self.hashF = self._difference_hash
        elif hash_function == "pHash":
            self.hashF = self._pHash
        else:
            raise ValueError

    def _difference_hash(self, image: np.ndarray, hashsize: int = 8) -> int:
        """
        Removes items from a list of images according to the class description

        Parameters
        -----------

        image: np.ndarray
            A numpy array that stores an image. The shape of the array is expected to be (h, w)
        
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
    
    def _pHash(self, image: np.ndarray, hashsize: int = 8):
        N = 32
        resizedImg = cv.resize(image, (N, N)).astype(np.float32)

        transformedImage = cv.dct(resizedImg)
        hashData = transformedImage[:hashsize, :hashsize]

        median = np.median(hashData)
        hashBits = hashData.flatten() > median

        hash = sum([2**i for (i, b) in enumerate(hashBits) if b])
        return hash

    def _hamming_distance(self, x: int, y: int):
        dist = bin(x ^ y).count("1")
        
        return dist
        

    def _load_images(self, strBatch: list[str]) -> np.ndarray:
        images = []

        for path in strBatch:
            img = Image.open(path).convert("L")
            img_arr = np.array(img).astype(np.uint8)
            images.append(img_arr)
        
        return images
    
    def split_hash(self, hash: int, prefix_bits: int = 32):
        hashSize = 64
        suffixBits = hashSize - prefix_bits

        prefix = hash >> suffixBits
        suffix = hash & ((1 << suffixBits) - 1)

        return prefix, suffix

    def find_duplicates(self, images: list[str], min_distance: int = 10, batch_size: int = 128) -> list[str]:
        """
        Finds duplicate images in a list of image paths and returns the paths of duplicate and similar images. The caller 
        is expected to execute the final removal of the duplicates himself.

        Parameters
        -----------

        images: list[str]
            A list containing all the image paths to be checked.
        
        min_distance: int, optional
            The minimum distances two images have to exceed to be considered different. The default value of 10 was 
            used in an article implementing a similar algorithm.

        batch_size: int, optional
            The number of images to load in one pass. If this number is set to high the execution will fail due to 
            high memory consumption.

        Returns
        --------

        duplicates: list[str]
            A list containing the paths of the filtered images. 
        """

        bucketMap: dict[int, list[tuple[int, str]]] = {}
        
        numImages = len(images)
        numPasses = math.ceil(numImages / batch_size)
        
        for i in range(numPasses):
            strBatch = images[i*batch_size: (i+1)*batch_size]
            imgBatch = self._load_images(strBatch=strBatch)
            hashes = list(map(self.hashF, imgBatch))

            entries = list(zip(hashes, strBatch))
            
            for hash, path in entries:
                prefix, suffix = self.split_hash(hash)

                f = lambda x: self._hamming_distance(suffix, x[0]) > min_distance

                index = prefix
                similar = bucketMap.setdefault(index, [])

                distancesOk = list(map(f, similar))
                if all(distancesOk) or len(similar) == 0:
                    similar.append((suffix, path))
        
        allLists = [x for lst in bucketMap.values() for x in lst]
        _, paths = zip(*allLists)
        unique = list(paths)

        duplicates = list(filter(lambda x: x not in unique, images))

        return duplicates

