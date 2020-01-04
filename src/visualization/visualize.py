import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_sample(eeg, img, x=260, y=260):
    plt.figure(figsize=(15 ,6))
    plt.suptitle("Sample data" ,fontsize = 20)
    plt.subplot(1 ,2 ,1)
    plt.title("EEG Heatmap" ,size=14)

    eeg_train = cv2.resize(np.array(eeg) ,(x, y))
    plt.imshow(eeg_train ,cmap=plt.cm.binary)
    plt.colorbar()

    plt.subplot(1 ,2 ,2)
    plt.title("Corresponding image" ,size = 14)

    plt.imshow(img,cmap=plt.cm.binary) # normalize image data
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pass