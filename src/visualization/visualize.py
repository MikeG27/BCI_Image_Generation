import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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
    if "val_loss" in history.history:
            ax.plot(history.history['val_loss'])
            plt.legend(['Train', 'Validation'], loc='upper right')
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    plt.grid()

    return fig



if __name__ == "__main__":
    pass


def evaluate_visually(model, X, y, rows=2, cols=6):
    fig = plt.figure(figsize=(14, 6))

    # cols = 6
    # rows = 2

    for i in range(1, cols * rows + 1):
        if i <= cols:

            fig.add_subplot(rows, cols, i)
            img = y[i].reshape(30, 30)
            plt.imshow(img, cmap=plt.cm.binary)
            plt.title("Stimulus")

        else:

            fig.add_subplot(rows, cols, i)
            x = X[i - cols].reshape(1, 30, 30, 1)
            x_test_encoded = model.predict(x, batch_size=1).reshape(30, 30)
            plt.imshow(x_test_encoded, cmap=plt.cm.binary)

            plt.title("From Brain ")

    return fig