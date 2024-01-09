"""
this module is used to detect faces in an image using haar cascade classifier
main goal if to detect faces in an image and then use the face to classify the image
"""
import cv2
import multiprocessing as mp
from tqdm import tqdm
from datamanagement.data_holder import DataHolder
import json

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('../data/.xml')


def detect_face(path):
    base_img = cv2.imread(path)
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    sclaf = 1.1
    min_neigh = 3
    flag = 0
    faces = face_cascade.detectMultiScale(gray, sclaf, min_neigh, flags=flag)
    return faces


def detect_facze(base_img):
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    sclaf = 1.8
    min_neigh = 1
    faces = face_cascade.detectMultiScale(gray, sclaf, min_neigh)
    return faces


if __name__ == '__main__':
    data = DataHolder(None, None, update=False)
    data.balanced_sample(1)
    x_train, y_train, x_test, y_test = data.give_me_my_data(data_type="RGB")

    # show the image
    for i in range(10):
        try:
            base_img = cv2.imread(data.train_df["path"].tolist()[i])
            faces = detect_facze(base_img)
            print(faces)
            if isinstance(faces, tuple):
                continue
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(base_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            x1, y1, x2, y2 = eval(data.train_df["box"].iloc[i])
            cv2.rectangle(base_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('imgs', base_img)
            cv2.waitKey(0)
        except Exception as e:
            print(e)
