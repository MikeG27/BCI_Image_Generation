import pandas as pd

def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize float columns in pandas dataframe
    """
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def memory_usage(df: pd.DataFrame) -> None :
    """Print memory usage of dataframe in Mb"""
    print(f"Memory usage {round(df.memory_usage().sum() / round(1000000),2)} Mb")