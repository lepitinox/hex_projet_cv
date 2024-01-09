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

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential

tqdm.pandas()
from datamanagement.data_holder import DataHolder

import multiprocessing as mp


def rgb_histogram(image):
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()


def run(df):
    ret = []
    for i in df['path'].tolist():
        img = cv2.imread(i)
        hist = rgb_histogram(img)
        ret.append(hist)
    return ret


if __name__ == '__main__':
    # clean the data
    data = DataHolder(train_df, test_df, update=False)
    x_train, y_train, x_test, y_test = data.give_me_my_data(data_type="RGB")
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = run(x_train)
    X_test = run(x_test)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train[0].shape,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    hist = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)

    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=data.classes))

