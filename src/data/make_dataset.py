import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.datasets import mnist

import config


def generate_image_dataset(X, y, output_dir, width=300, height=300, cmap="gray"):
    """
    Generate 300x300 gray images from mnist dataset and extract images to train/test folder
    :param X: numpy images list
    :param y: numpy class list
    :param output_dir: output directory for generated images
    :return:
    """

    if len(os.listdir(output_dir)) != 0 :
        print(f"Directory {output_dir} is not empty")
        return

    dim = (width, height)
    for i in tqdm(range(len(X))):
        numpy_img = X[i]
        image_number = y[i]
        resized_img = cv2.resize(numpy_img, dim, interpolation=cv2.INTER_AREA)
        plt.imsave(f"{output_dir}/image{i}_number{image_number}.png", resized_img, cmap="gray")

def generate_image_csv():
    pass

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    generate_image_dataset(x_train, y_train, output_dir=config.IMG_TRAIN_RAW_DIR)
    generate_image_dataset(x_test, y_test, output_dir=config.IMG_TEST_RAW_DIR)

    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

if __name__ == '__main__':
    main()
