import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns

def plot_sample(eeg, img, x=260, y=260):
    # ??????
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


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

def plot_sensors_correlation(df,figsize):

    """
    Plot corelation matrix based on Pearson and Spearman indicator
    :param df:
    :param figsize:
    :return:
    """

    figure, axes = plt.subplots(1, 2,sharex=False,figsize=figsize)

    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")

    #############Pearson corelation##################

    # Generate mask
    mask_pearson = generate_mask(pearson)
    mask_spearman = generate_mask(spearman)

    plot_heatmap(pearson,mask_pearson,axes[0])
    plot_heatmap(spearman, mask_spearman, axes[1])

    return pearson,spearman


def generate_mask(corr):
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    return mask


def plot_heatmap(corr, mask, ax):
    heatmap = sns.heatmap(corr,
                          square=False,
                          mask=mask,
                          linewidths=0.1,
                          cmap="coolwarm",
                          annot=True,
                          ax=ax,
                          cbar=False)

    _ = heatmap.set_title("Spearman Corealation", size=20)


def plot_learning_curve(history, figsize=(15, 5)):
    fig,ax = plt.subplots(1,1,figsize=figsize)
    # Plot training & validation loss values
    ax.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    plt.grid()
    # plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()

    return fig



if __name__ == "__main__":
    pass