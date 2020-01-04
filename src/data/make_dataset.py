import os

import cv2
import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

import config


def generate_image_dataset(X, width=300, height=300):
    """
    Generate 300x300 gray images from mnist dataset and extract images to train/test folder
    :param X: numpy images list
    :param y: numpy class list
    :param output_dir: output directory for generated images
    :return:

    """

    image_list = []

    dim = (width, height)
    for i in tqdm(range(len(X))):
        numpy_img = X[i]
        resized_img = cv2.resize(numpy_img, dim, interpolation=cv2.INTER_AREA)
        image_list.append(resized_img)

    return np.array(image_list)


def save_generated_array(array, output_dir, output_name):
    """
    Save array to a binary file in numpy

    :param array: image array
    :param output_dir:  output directory
    :param output_name : name of the file

    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(output_dir, output_name)
    np.save(output_path, array)

    print(f"[{output_name}] numpy array was saved into {output_path}\n")


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape dataset
    x_train = generate_image_dataset(x_train,output_dir=config.RAW_IMG_TRAIN_DIR)
    x_test = generate_image_dataset(x_test,output_dir=config.RAW_IMG_TEST_DIR)

    # save dataset
    print("Saving image data ....")
    save_generated_array(x_train, config.PREPROCESSED_IMG_TRAIN_DIR, "X_train")
    save_generated_array(y_train, config.PREPROCESSED_IMG_TRAIN_DIR, "y_train")
    save_generated_array(x_test, config.PREPROCESSED_IMG_TEST_DIR, "X_test")
    save_generated_array(y_test, config.PREPROCESSED_IMG_TEST_DIR, "y_test")


if __name__ == '__main__':
    main()
