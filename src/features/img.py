import numpy as np
import cv2

def normalize(img):
    return img.astype("float") /255/0

def resize(img,x,y):
    return cv2.resize(np.array(img),(x, y))

if __name__ == "__main__":
    pass