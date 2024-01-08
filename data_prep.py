import pandas as pd
from data_loader import train_df


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


if __name__ == "__main__":
    print(f" random sample of 5 rows: {train_df['box'].sample(5)}")
    train_df = clean_bbox(train_df)
    print(f" random sample of 5 rows: {train_df['box'].sample(5)}")
    print(f"Checking problematic rows: {train_df.iloc[901]}")
