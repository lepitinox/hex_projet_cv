"""
This module contains the data holder class
"""
from pathlib import Path

import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datamanagement.data_loader import train_df, test_df

matplotlib.use('TkAgg')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .data_prep import clean_bbox, change_path, find_shape, canals
from .data_loader import train_df, test_df


class DataHolder:
    """
    This class is used to hold the dataframes
    """
    base_path = Path("D:/pycharm/hex_projet_cv/data/")
    target_size = (64, 64)
    batch_size = 32

    def __init__(self, train_df=train_df, test_df=test_df, update=True):
        """
        Constructor of the DataHolder class
        Parameters
        ----------
        train_df : pd.DataFrame
            Dataframe containing the training data
        test_df : pd.DataFrame
            Dataframe containing the test data
        """
        self.le = LabelEncoder()
        if update:
            self.train_df = train_df
            self.test_df = test_df
            self._clean_data()
            self.enrich_train_df()
            self.enrich_test_df()
            self.save_train_df()
            self.save_test_df()
        else:
            self.train_df = self.load_train_df()
            self.test_df = self.load_test_df()

        self.train_generator = None
        self.validation_generator = None
        self.create_data_pipline()

    def _clean_data(self):
        """
        This function cleans the data
        Returns
        -------
        pd.DataFrame
            cleaned dataframe
        """
        self.train_df = clean_bbox(self.train_df)
        self.train_df = change_path(self.train_df, self.base_path / "train")
        self.test_df = clean_bbox(self.test_df)
        self.test_df = change_path(self.test_df, self.base_path / "test")

    def enrich_train_df(self):
        """
        This function enriches the train_df with functions from data_prep
        Parameters
        ----------
        train_df : pd.DataFrame
            Dataframe to add to the train_df
        """
        self.train_df = find_shape(self.train_df)
        self.train_df = canals(self.train_df)

    def enrich_test_df(self):
        """
        This function enriches the test_df with functions from data_prep
        Parameters
        ----------
        test_df : pd.DataFrame
            Dataframe to add to the test_df
        """
        self.test_df = find_shape(self.test_df)
        self.test_df = canals(self.test_df)

    def save_train_df(self):
        """
        This function saves the train_df
        """
        self.train_df.to_csv(self.base_path / "train_df.csv", index=False)

    def save_test_df(self):
        """
        This function saves the test_df
        """
        self.test_df.to_csv(self.base_path / "test_df.csv", index=False)

    def load_train_df(self):
        """
        This function loads the train_df
        """
        return pd.read_csv(self.base_path / "train_df.csv")

    def load_test_df(self):
        """
        This function loads the test_df
        """
        return pd.read_csv(self.base_path / "test_df.csv")

    def label_enc(self):
        """
        This function encodes the labels
        """
        train_labels = self.le.fit_transform(self.train_df["label"]).astype(str)
        test_labels = self.le.transform(self.test_df["label"]).astype(str)
        return train_labels, test_labels

    def sample(self, pct=0.1):
        """
        This function samples the train_df and the test_df
        """
        self.reaload_data()
        self.train_df = self.train_df.sample(frac=pct)
        self.test_df = self.test_df.sample(frac=pct)

    def give_me_my_data(self, data_type="all"):
        if data_type == "RGB":
            self.train_df = self.train_df[self.train_df["canals"] == "RGB"]
            self.test_df = self.test_df[self.test_df["canals"] == "RGB"]
        elif data_type == "grayscale":
            self.train_df = self.train_df[self.train_df["canals"] == "grayscale"]
            self.test_df = self.test_df[self.test_df["canals"] == "grayscale"]
        elif data_type == "all":
            pass
        y_train, y_test = self.label_enc()
        x_train = self.train_df.drop(columns=["label"], axis=1)
        x_test = self.test_df.drop(columns=["label"], axis=1)
        return x_train, y_train, x_test, y_test

    def reaload_data(self):
        self.train_df = self.load_train_df()
        self.test_df = self.load_test_df()

    def create_generators(self):
        self.train_generator = ImageDataGenerator(rescale=1. / 255,
                                                  shear_range=0.2,
                                                  zoom_range=0.2,
                                                  horizontal_flip=True)
        self.validation_generator = ImageDataGenerator(rescale=1. / 255)

    def create_data_pipline(self):
        self.create_generators()
        self.train_generator = self.train_generator.flow_from_dataframe(dataframe=self.train_df,
                                                                        directory=self.base_path / "train",
                                                                        x_col="path",
                                                                        y_col="label",
                                                                        class_mode="categorical",
                                                                        target_size=self.target_size,
                                                                        batch_size=self.batch_size)

        # This is a similar generator, for validation data
        self.validation_generator = self.validation_generator.flow_from_dataframe(dataframe=self.test_df,
                                                                                  directory=self.base_path / "test",
                                                                                  x_col="path",
                                                                                  y_col="label",
                                                                                  class_mode="categorical",
                                                                                  target_size=self.target_size,
                                                                                  batch_size=self.batch_size)
