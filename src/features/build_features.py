import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, quantile_transform
from sklearn.preprocessing import PowerTransformer


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


def quantile_transformer(df,output_distribution = "normal"):
    """

    :param df:
    :return:
    """
    columns = df.columns
    X = quantile_transform(df, output_distribution=output_distribution, random_state=0, copy=True)
    df = pd.DataFrame(X)
    df.columns = columns
    return df


def transform_distribution(df, method="yeo-johnson"):
    #TEGO NIE WYKORZYSTUJE
    pt = PowerTransformer(method=method, standardize=False)
    df_gausian = pt.fit_transform(df)
    return df_gausian, pt


if __name__ == "__main__":
    pass
