"""
This module classifies the images using HOG
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from tqdm import tqdm

from data_loader import train_df, test_df
from data_holder import DataHolder

import multiprocessing as mp

from skimage.feature import hog

def create_histogram(path):
    image = cv2.imread(path)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    return fd

if __name__ == '__main__':
    # clean the data
    data = DataHolder(train_df, test_df, update=False)
    x_train, y_train, x_test, y_test = data.give_me_my_data(data_type="RGB")
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    paths = x_train['path'].tolist()

    # create the histogram
    if not Path("data/Xtrain_hog.parquet").exists():
        with mp.Pool() as p:
            hist = list(tqdm(p.imap(create_histogram, paths), total=len(paths)))
        # create the dataframe
        Xtrain = pd.DataFrame({"hog": hist, "label": y_train})
        Xtrain.to_parquet("data/Xtrain_hog.parquet")

        with mp.Pool() as p:
            hist = list(tqdm(p.imap(create_histogram, x_test['path'].tolist()), total=len(x_test['path'].tolist())))
        Xtest = pd.DataFrame({"hog": hist, "label": y_test})
        Xtest.to_parquet("data/Xtest_hog.parquet")
    else:
        Xtrain = pd.read_parquet("data/Xtrain_hog.parquet")
        Xtest = pd.read_parquet("data/Xtest_hog.parquet")
    model = SVC()
    model.fit(Xtrain["hog"], y_train)
    le = data.le
    predictions = model.predict(Xtest["hog"])
    print(classification_report(y_train, predictions, target_names=le.classes_))

