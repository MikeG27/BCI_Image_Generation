import argparse
import os

import cv2
import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, quantile_transform
from sklearn.preprocessing import PowerTransformer

# path
import sys

sys.path.append(os.getcwd())

import config
from config import DATA_PREPROCESSED_DIR
from config import train_ratio, test_ratio, validation_ratio
from src.utils.preprocessing import save_preprocessing_object


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', type=str, default=config.GDRIVE_FILE,
                        help="csv file in raw folder")

    parser.add_argument('-s', type=float, nargs="+", default=(train_ratio, test_ratio, validation_ratio),
                        help='Select data split ratio'
                             '1 index - train ratio\n'
                             '2 index - test ratio\n'
                             '3 index - valid ratio')

    args = parser.parse_args()

    return args


def check_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().mean() * 100).sort_values(ascending=False)
    dtype = df.dtypes
    missing_stats = pd.concat([total, percent, dtype], axis=1, keys=['Total', 'Percent', "Dtype"])
    return missing_stats


def check_sampling(df, sampling):
    """
    Return list of dataframe samples where sampling is incorrect
    """
    feedback = {}
    n_index = df.mnist_index.nunique()
    for index in range(n_index):
        sample_len = len(df[df.mnist_index == index].mnist_index)  # filtering
        if sample_len != sampling:
            feedback[str(index)] = sample_len

    return feedback


def resize(eeg, x, y):
    new_shape = (x, y)
    eeg_resized = cv2.resize(np.array(eeg), new_shape)
    return eeg_resized


def drop_missing(df):
    # maybe based on missing values percentage?
    if df.isna().sum().sum() != 0:
        return df.dropna()
    else:
        return df


def get_outliers_iqr(df, iqr_mul=3):
    """
    Return upper and lower bound of outliers from pandas dataframe based on IQR indicator.
    :param df: Pandas dataFrame
    :param IQR_mul : IQR_mult > 1.5 - normal outliers, IQR_mul > 3 - extreme outliers
    :return df_out :
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    lower_outliers = df < (Q1 - iqr_mul * IQR)
    upper_outliers = df > (Q3 + iqr_mul * IQR)

    return lower_outliers, upper_outliers


def normalize_min_max(df):
    """

    :param df:
    :return:it
    """
    columns = df.columns

    scaler = MinMaxScaler().fit(df)
    data_normalized = scaler.transform(df)
    df_normalized = pd.DataFrame(data_normalized)
    df_normalized.columns = columns
    return df_normalized, scaler


def quantile_transformer(df, output_distribution="normal"):
    """

    :param df:
    :return:
    """
    columns = df.columns
    X = quantile_transform(df, output_distribution=output_distribution, random_state=0, copy=True)
    df = pd.DataFrame(X)
    df.columns = columns
    return df


# def transform_distribution(df, method="yeo-johnson"):
#     # TEGO NIE WYKORZYSTUJE
#     pt = PowerTransformer(method=method, standardize=False)
#     df_gausian = pt.fit_transform(df)
#     return df_gausian, pt


def batch_data():
    mnist_indexes = df.mnist_index.value_counts()
    n_mnist_indexes = len(mnist_indexes)
    eeg_batched = [df[df["mnist_index"] == index][sensors_list].values for index in range(n_mnist_indexes)]
    images_batched = [X_train[i] for i in range(n_mnist_indexes)]
    # return transformed dataframe
    df_batched = pd.DataFrame({"eeg": eeg_batched, "img": images_batched})

    return df_batched


def train_test_validation_split(X, y, train_ratio, test_ratio, validation_ratio):
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

    # val-test split
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))
    return X_train, X_test, X_val, y_train, y_test, y_val


if __name__ == "__main__":
    # Parse data

    args = parser()

    filename = args.f
    train_ratio_x = args.s[0]
    test_ratio_x = args.s[1]
    validation_ratio_x = args.s[2]

    print(train_ratio)
    print(test_ratio)

    print(f"Train ratio : {train_ratio}")
    print(f"Test ratio : {test_ratio}")
    print(f"Validation ratio : {validation_ratio}\n")


    # ******************* EEG *************************

    print("Read data....")
    csv_path = os.path.join(config.RAW_EEG_DIR, filename)
    df = pd.read_csv(csv_path, index_col=0)

    # drop mnist_class
    df.drop(labels="mnist_class", axis=1, inplace=True, errors="ignore")
    sensors_list = df.columns[:14]

    print("Make Gaussian....")
    #df[sensors_list] = quantile_transformer(df[sensors_list])

    print("Scale eeg...")
    df[sensors_list], scaler = normalize_min_max(df[sensors_list])

    print("Save scaller....")
    scaling_object = scaler
    filepath = os.path.join(config.MODEL_DIR, "normalizer.h5")
    save_preprocessing_object(scaling_object, filepath)

    # ******************* IMG *************************
    print("Scale images....")
    (X_train, y_train), (_, _) = mnist.load_data()
    # X_train = X_train[0:1202]
    # y_train = y_train[0:1202]
    X_train = X_train.astype("float") / 255.0  # IMG normalization

    # ******************* Data reshaping *************************

    print("Data reshape....")
    df_batched = batch_data()
    # resize with interpolation to prefered shape
    df_batched["eeg"] = df_batched["eeg"].apply(lambda x: cv2.resize(x, (30, 30)))
    df_batched["img"] = df_batched["img"].apply(lambda x: cv2.resize(x, (30, 30)))

    # Flatten to VAE Dense input
    #df_batched["eeg"] = df_batched["eeg"].apply(lambda x: x.flatten())
    #df_batched["img"] = df_batched["img"].apply(lambda x: x.flatten())

    print("Get features and labels....\n")

    X = df_batched.eeg.values
    y = df_batched.img.values

    # Squeeze data

    X = np.array([X[i] for i in range(len(X))]).squeeze()
    y = np.array([y[i] for i in range(len(X))]).squeeze()

    print(f"Features shape : {X.shape}")
    print(f"Outputs  shape : {y.shape}\n")

    # ***************** Train-test-validation-split **************************

    print("Train-test-valid split..... : \n")
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_validation_split(X, y, train_ratio=train_ratio_x
                                                                                 , test_ratio=test_ratio_x,
                                                                                 validation_ratio=validation_ratio_x)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"X_valid shape : {X_val.shape}")

    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")
    print(f"X_valid shape : {y_val.shape}")

    # Save to preprocessed data folder

    print(f"\nSave data to {DATA_PREPROCESSED_DIR} ..... : \n")

    np.save(os.path.join(config.DATA_PREPROCESSED_DIR, "X_train"), X_train)
    np.save(os.path.join(config.DATA_PREPROCESSED_DIR, "X_test"), X_test)
    np.save(os.path.join(config.DATA_PREPROCESSED_DIR, "X_valid"), X_val)

    np.save(os.path.join(config.DATA_PREPROCESSED_DIR, "y_train"), y_train)
    np.save(os.path.join(config.DATA_PREPROCESSED_DIR, "y_test"), y_test)
    np.save(os.path.join(config.DATA_PREPROCESSED_DIR, "y_valid"), y_val)

    print("Data preprocessing was ended")
