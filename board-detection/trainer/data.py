import os

import cv2 as cv

class Dataset:
    def __init__(self, path):
        self.train_path = os.path.join(path, "train")
        self.test_path = os.path.join(path, "test")
        self.valid_path = os.path.join(path, "valid")
    
    def get_train_image(self, name):
        image_path = os.path.join(self.train_path, "images", name)
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        assert img is not None, "file could not be read, check with os.path.exists()"
        return img