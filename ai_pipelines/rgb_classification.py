"""
This module is used to use the RGB histogram as a feature for classification
"""
from pathlib import Path

import matplotlib
import pandas as pd

from datamanagement.data_loader import train_df, test_df

matplotlib.use('TkAgg')
import numpy as np
import cv2
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

tqdm.pandas()
from datamanagement.data_holder import DataHolder

import multiprocessing as mp


def create_histogram(path):
    image = cv2.imread(path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


if __name__ == '__main__':
    # clean the data
    data = DataHolder(train_df, test_df, update=False)
    x_train, y_train, x_test, y_test = data.give_me_my_data(data_type="RGB")
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    paths = x_train['path'].tolist()

    # create the histogram
    if not Path("data/Xtrain_rgb.parquet").exists():
        with mp.Pool() as p:
            hist = list(tqdm(p.imap(create_histogram, paths), total=len(paths)))
        # create the dataframe
        Xtrain = pd.DataFrame({"rgb": hist, "label": y_train})
        Xtrain.to_parquet("data/Xtrain_rgb.parquet")

        with mp.Pool() as p:
            hist = list(tqdm(p.imap(create_histogram, x_test['path'].tolist()), total=len(x_test['path'].tolist())))
        Xtest = pd.DataFrame({"rgb": hist, "label": y_test})
        Xtest.to_parquet("data/Xtest_rgb.parquet")
    else:
        Xtrain = pd.read_parquet("data/Xtrain_rgb.parquet")
        Xtest = pd.read_parquet("data/Xtest_rgb.parquet")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(Xtrain["rgb"].values, y_train)
    le = data.le
    predictions = model.predict(Xtest["rgb"])
    print(classification_report(y_train, predictions, target_names=le.classes_))
