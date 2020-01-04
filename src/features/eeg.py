import os
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.externals import joblib

"""
"""


def read_data(path):
    """
    I assume that data comes from emotiv epoc+ so I select data by one scenario
    :param path:
    :return:
    """
    # check if path exist
    if os.path.exists(path):
        df = pd.read_csv(path, sep=",", parse_dates=[0], index_col=0)
        df.reset_index(inplace=True)
        df = df[df.columns[2:16]]
        return df
    else:
        print("EEG Data not found !")


def resize(eeg, x, y):
    #eeg sample not all df , np or cv2 ?
    new_shape = (x,y)
    eeg_resized = cv2.resize(np.array(eeg),new_shape)
    return eeg_resized

def check_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().mean() * 100).sort_values(ascending=False)
    dtype = df.dtypes

    missing_stats = pd.concat([total, percent, dtype], axis=1, keys=['Total', 'Percent', "Dtype"]).sort_values(
        "Percent", ascending=False)

    print("Missing values INFO :\n")
    print("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(
        df.shape[0]) + ' Rows.\n')

    return missing_stats


def drop_missing(df):
    # maybe based on missing values percentage?
    if df.isna().sum().sum() != 0 :
        return df.dropna()
    else :
        return df


def remove_outliers_iqr(df, iqr_mul=1.5):
    """
    Remove outliers from pandas dataframe based on IQR indicator.
    :param df: Pandas dataFrame
    :param iqr_mul : IQR_mult > 1.5 - normal preprocessing, IQR_mul > 3 - extreme preprocessing
    :return df_out :
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_outliers = df < (Q1 - iqr_mul * IQR)
    upper_outliers = df > (Q3 + iqr_mul * IQR)
    df_out = df[~((lower_outliers) | (upper_outliers)).any(axis=1)]
    print("Removed {} rows based on IQR".format(df.shape[0] - df_out.shape[0]))
    return df_out


def normalize(df):
    """

    :param df:
    :return:
    """
    columns = df.columns

    scaler = MinMaxScaler().fit(df)
    data_normalized = scaler.transform(df)
    df_normalized = pd.DataFrame(data_normalized)
    df_normalized.columns = columns
    return df_normalized, scaler


def transform_distribution(df,method = "yeo-johhson"):
    #Poprawic
    pt = PowerTransformer(method= method,standardize=False)
    pt = pt.fit(df)
    return pt


def save_preprocessing_object(object, filename):
    joblib.dump(object, filename)
    print(f"Object was saved into path : {filename}")


def load_preprocessing_object(filename):
    object = joblib.load(filename)
    return object


if __name__ == "__main__":
    pass
