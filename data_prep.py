import pandas as pd
from matplotlib import pyplot as plt

from data_loader import train_df
from tqdm import tqdm

tqdm.pandas()


def clean_bbox(df: pd.DataFrame, bbox_column: str = 'box') -> pd.DataFrame:
    """
    This function cleans the bbox column of the dataframe
    Expecting a list of 4 values
    bbox_column may contain multiple whitespace between values, . and values are separated by spaces
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to clean
    bbox_column : str
        column name of the bbox column

    Returns
    -------
    pd.DataFrame
        cleaned dataframe
    """
    # TODO: optimize this function
    df[bbox_column] = df[bbox_column].apply(lambda x: x.replace('[', ''))
    df[bbox_column] = df[bbox_column].apply(lambda x: x.replace(']', ''))
    df[bbox_column] = df[bbox_column].apply(lambda x: x.split(' '))
    df[bbox_column] = df[bbox_column].apply(lambda x: [i.replace(" ", "").strip(".") for i in x])
    df[bbox_column] = df[bbox_column].apply(lambda x: [int(float(i)) for i in x if i != ''])
    return df


def change_path(df, path='data/train'):
    """
    This function changes the path of the images
    Base path point to the folder Archive, we want to point to the folder train
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to clean

    Returns
    -------
    pd.DataFrame
        cleaned dataframe
    """
    df['path'] = df['path'].apply(lambda x: x.replace('/Archive', path))
    return df


def find_shape(df):
    """
    This function finds the shape of the images
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to clean

    Returns
    -------
    pd.DataFrame
        cleaned dataframe
    """
    df['shape'] = df['path'].progress_apply(lambda x: plt.imread(x).shape)
    return df


def canals(df):
    """
    This function finds the number of canals of the images
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to clean

    Returns
    -------
    pd.DataFrame
        cleaned dataframe
    """
    canals_dict = {1: 'grayscale', 3: 'RGB', 4: 'RGBA'}
    df['canals'] = df['shape'].apply(lambda x: canals_dict[x[2]] if len(x) == 3 else canals_dict[1])
    return df


if __name__ == "__main__":
    print(f" random sample of 5 rows: {train_df['box'].sample(5)}")
    train_df = clean_bbox(train_df)
    train_df = change_path(train_df)
    print(f" random sample of 5 rows: {train_df['box'].sample(5)}")
    print(f"Checking problematic rows: {train_df.iloc[901]}")
