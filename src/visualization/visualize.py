import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib import pyplot as plt

# path
import sys
sys.path.append(os.getcwd())

import config


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', type=str, default=config.GDRIVE_FILE,
                        help="csv file in raw folder")
    args = parser.parse_args()

    return args


def set_figure_size(df):
    n_features = df.shape[1]
    figsize = (n_features*1.2,n_features/2)
    return figsize

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

    methods = ["pearson","spearman"]

    pearson = df.corr(method=methods[0])
    spearman = df.corr(method=methods[1])

    #############Pearson corelation##################

    # Generate mask
    pearson_mask = generate_mask(pearson)
    spearman_mask = generate_mask(spearman)

    build_heatmap(pearson, pearson_mask,axes[0],methods[0])
    build_heatmap(spearman, spearman_mask,axes[1],methods[1])

    return figure


def generate_mask(corr):
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    return mask

def build_heatmap(corr, mask,ax,method):
    heatmap = sns.heatmap(corr,
                          square=False,
                          mask=mask,
                          linewidths=0.1,
                          cmap="coolwarm",
                          annot=True,
                          ax = ax,
                          cbar=False)

    _ = heatmap.set_title(f"{method} corealation", size=20)



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


def evaluate_visually(model, X, y, rows=2, cols=6):
    fig = plt.figure(figsize=(14, 6))

    for i in range(1, cols * rows + 1):
        if i <= cols:

            fig.add_subplot(rows, cols, i)
            img = y[i].reshape(config.image_shape[0], config.image_shape[1])
            plt.imshow(img, cmap=plt.cm.binary)
            plt.title("Stimulus")

        else:

            fig.add_subplot(rows, cols, i)
            x = X[i - cols].reshape(1, config.image_shape[0], config.image_shape[1], 1)
            x_test_encoded = model.predict(x, batch_size=1).reshape(config.image_shape[0], config.image_shape[1])
            plt.imshow(x_test_encoded, cmap=plt.cm.binary)

            plt.title("From Brain")

    return fig



if __name__ == "__main__":

    print("[RUN ANALYTICS..]")
    # Parse arguments
    args = parser()

    #load dataframe
    filename = args.f
    csv_path = os.path.join(config.RAW_EEG_DIR, filename)
    df = pd.read_csv(csv_path, index_col=0)

    # drop mnist_class
    try :
        df.drop(labels="mnist_class", axis=1, inplace=True, errors="ignore")
    except :
        print("mnist_class not_exist")

    # get sensors values
    sensors_list = df.columns[0:-1] #without mnist_index
    df_sensors = df[sensors_list]

    # set figure size
    figsize = set_figure_size(df_sensors)

    # # PLOT
    plt.figure(figsize=figsize)
    _ = df_sensors.plot(figsize=figsize)
    _ = plt.xlabel("Samples")
    _ = plt.ylabel("Voltage")
    _ = plt.title("EEG characteristics")
    plt.savefig(config.FIGURES_PLOT)

    # # BOXPLOT
    plt.figure(figsize=figsize)
    _ = df_sensors.boxplot(figsize=figsize)
    _ = plt.title("EEG boxplot")
    plt.savefig(config.FIGURES_BOXPLOT)

    # DISTRIBUTIONS
    plt.figure(figsize=figsize)
    _ = df_sensors.hist(figsize=figsize,bins=df_sensors.shape[1]*2)
    _ = plt.title("EEG histogram")
    plt.savefig(config.FIGURES_DISTRIBUTIONS)

    # # PDF
    plt.figure(figsize=figsize)
    for i, sensor in enumerate(df_sensors.columns):
        _ = sns.distplot(df_sensors[sensor], hist=False, label=sensor)
    _ = plt.title("Probability Density Function", size=15)
    _ = plt.xlabel("Voltage")
    _ = plt.ylabel("Probability")
    plt.savefig(config.FIGURES_PDF)

    # #ECDF
    plt.figure(figsize=figsize)
    for i, sensor in enumerate(df_sensors.columns):
        x, y = ecdf(df_sensors[sensor])
        plt.plot(x, y, label=sensor)

    _ = plt.title("Empirical Cumulative Density Function ", size=15)
    _ = plt.xlabel("Voltage")
    _ = plt.ylabel("Fraction of Data")

    plt.savefig(config.FIGURES_ECDF)

    # #CORRELATION (SPEARMAN & PEARSON)
    figure = plot_sensors_correlation(df_sensors, figsize=figsize)
    plt.colorbar()
    plt.savefig(config.FIGURES_CORR)

