"""
This module contains the data holder class
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_prep import clean_bbox, change_path, find_shape

class DataHolder:
    """
    This class is used to hold the dataframes
    """
    base_path = "D:\\pycharm\\hex_projet_cv\\data\\"

    def __init__(self, train_df, test_df, update=True):
        """
        Constructor of the DataHolder class
        Parameters
        ----------
        train_df : pd.DataFrame
            Dataframe containing the training data
        test_df : pd.DataFrame
            Dataframe containing the test data
        """
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
        self.le = LabelEncoder()
        self.trainloader = None
        self.testloader = None


    def _clean_data(self):
        """
        This function cleans the data
        Returns
        -------
        pd.DataFrame
            cleaned dataframe
        """
        self.train_df = clean_bbox(self.train_df)
        self.train_df = change_path(self.train_df, self.base_path + "train")
        self.test_df = clean_bbox(self.test_df)
        self.test_df = change_path(self.test_df, self.base_path + "test")

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
        self.train_df.to_csv(self.base_path + "train_df.csv", index=False)

    def save_test_df(self):
        """
        This function saves the test_df
        """
        self.test_df.to_csv(self.base_path + "test_df.csv", index=False)

    def load_train_df(self):
        """
        This function loads the train_df
        """
        return pd.read_csv(self.base_path + "train_df.csv")

    def load_test_df(self):
        """
        This function loads the test_df
        """
        return pd.read_csv(self.base_path + "test_df.csv")

    def label_enc(self):
        """
        This function encodes the labels
        """
        train_labels = self.le.fit_transform(self.train_df["label"])
        test_labels = self.le.transform(self.test_df["label"])
        return train_labels, test_labels

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

