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
    data.balanced_sample(1)
    x_train, y_train, x_test, y_test = data.give_me_my_data(data_type="RGB")
    if multipro:
        with mp.Pool() as p:
            faces = list(tqdm(p.imap(detect_face, x_train["path"].tolist()), total=len(x_train["path"].tolist())))
    else:
        faces = []
        for i in tqdm(x_train["path"].tolist()):
            faces.append(detect_face(i))
    try:
        data.train_df["faces"] = str(faces)
    except Exception as e:
        json.dump(faces, open("data/faces.json", "w"))

    # show the image
    for i in range(10):
        try:
            base_img = cv2.imread(data.train_df["path"].tolist()[i])
            print(faces[i])
            for listodl in faces:
                if listodl == ():
                    continue
                for face in listodl[0]:
                    x, y, w, h = face
                    cv2.rectangle(base_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('img', base_img)
            cv2.waitKey(0)
        except Exception as e:
            print(e)

