import argparse
import time

import cv2
from keras.datasets import mnist

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1200,
                        help="How many images do you want?")
    parser.add_argument("-d", type=float, default=5,
                        help="How long delay you want before start?")

    parser.add_argument("-d2", type=float, default=1,
                        help="How long delay you want between images?")

    args = parser.parse_args()
    experiment(args)


def experiment(args):
    n_images = args.n
    start_delay = args.d
    image_delay = args.d2

    (x_train, _), (_, _) = mnist.load_data()
    X = x_train[0:n_images]

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i in range(len(X)):
        frame = X[i]
        if i == 0:
            time.sleep(start_delay)
        cv2.imshow('frame', frame)
        time.sleep(image_delay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Experiment was ended")


if __name__ == "__main__":
    parser()
