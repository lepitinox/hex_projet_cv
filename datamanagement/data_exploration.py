import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
from data_loader import train_df, test_df

print(train_df.shape)
print(test_df.shape)


def explore_dataframes():
    print("Train dataset:")
    print(f"Dataset is {train_df.shape[0]} rows by {train_df.shape[1]} columns")
    print(f"Columns are : '{', '.join(list(train_df.columns))}'")
    print(f"Types of columns: {train_df.dtypes}")
    print(f"Columns example: {train_df.iloc[0]}")
    print("histogram of the log of size:")
    # sns.histplot(train_df['size'], bins=100, log_scale=True)
    # plt.show()
    print(train_df.describe())
    print("Data are mostly of the same size, but there are some outliers")
    print(f"Extension(s) of images: {train_df['extension'].unique()}")

    print(f" random sample of 5 rows: {train_df['box'].sample(5)}")
    print("box data have a bad quality, we will need to clean them")

    print(f"Labels: {train_df['label'].unique()}")
    print(f"Labels count: {train_df['label'].value_counts()}")
    print(f"data are heavily unbalanced")
    print(f"Labels pct: {(train_df['label'].value_counts() / train_df.shape[0] * 100).round(3)}")
    print(f"Nans: {train_df.isna().sum()}")
    # plot histogram of labels proportion
    sns.histplot(train_df['label'])
    plt.xticks(rotation=45, ha="right")
    plt.show()
    sns.histplot(test_df['label'])
    plt.xticks(rotation=45, ha="right")
    plt.show()


    image_path = Path('../data/train')

    print(f"Number of images: {len(list(image_path.glob('**/*.jpg')))}")


def explore_images():
    images_path = Path('../data/train')
    images = list(images_path.glob('**/*.jpg'))
    print(f"Number of images: {len(images)}")
    print(f"Example of image path: {images[0]}")
    print("showing random sample of 5 images:")
    fig, ax = plt.subplots(1, 5, figsize=(20, 20))
    for i in range(5):
        ax[i].imshow(plt.imread(images[i]))
        ax[i].set_title(images[i].name)
    plt.show()
    print("in somme images there are multiple people, we will need to clean the data")
    print("but also some images there are no faces, we will need to clean the data")


if __name__ == '__main__':
    explore_dataframes()
    explore_images()
