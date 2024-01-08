"""
this module is used to detect faces in an image using haar cascade classifier
main goal if to detect faces in an image and then use the face to classify the image
"""
import cv2
import multiprocessing as mp
from tqdm import tqdm
from datamanagement.data_holder import DataHolder

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')


def detect_face(path):
    base_img = cv2.imread(path)
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    sclaf = 1.4
    min_neigh = 4
    faces = face_cascade.detectMultiScale(gray, sclaf, min_neigh)
    return faces


if __name__ == '__main__':
    multipro = True
    data = DataHolder(None, None, update=False)
    x_train, y_train, x_test, y_test = data.give_me_my_data(data_type="RGB")
    if multipro:
        with mp.Pool() as p:
            faces = list(tqdm(p.imap(detect_face, x_train["path"].tolist()), total=len(x_train["path"].tolist())))
    else:
        faces = []
        for i in tqdm(x_train["path"].tolist()):
            faces.append(detect_face(i))
    print(faces)
