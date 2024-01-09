# load libraries
import uuid

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from datamanagement.data_holder import DataHolder
from tqdm import tqdm
import pandas as pd

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)


def find_face(img_path):
    imgs = Image.open(img_path)
    output = model(imgs)
    results = Detections.from_ultralytics(output[0])
    return results

def metric(bbox,faces):
    score = 0
    x, y, x2, y2 = bbox
    for face in faces.xyxy:
        face = map(int, face)
        x_f, y_f, x2_f, y2_f = face
        if x_f <= x <= x2_f and y_f <= y <= y2_f:
            score += 10
        else:
            score -= 1
    return score *2 / len(faces.xyxy)


if __name__ == '__main__':
    data = DataHolder(None, None, update=False)
    data.balanced_sample(1)
    data.test_balanced_sample(1)
    x_train, y_train, x_test, y_test = data.give_me_my_data()
