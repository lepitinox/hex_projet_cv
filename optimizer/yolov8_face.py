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


def yolo_optimze_add_c(df) -> pd.DataFrame:
    data = {}
    for i,j in enumerate(tqdm(df["path"].tolist())):
        faces = find_face(j)
        label = df.iloc[i]["label"]
        for face in faces.xyxy:
            face = map(int, face)
            x, y, x2, y2 = face
            # crop the image
            img = cv2.imread(j)
            crop_img = img[y:y2, x:x2]
            if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
                continue
            # save the cropped image with the name of the base image label and unique id
            cv2.imwrite(f"D:/pycharm/hex_projet_cv/data/train/custom_data/{label}_{uuid.uuid4()}.jpg", crop_img)
            data[f"{label}_{uuid.uuid4()}.jpg"] = [label, f"{label}_{uuid.uuid4()}.jpg"]
    new_df = pd.DataFrame.from_dict(data, orient="index", columns=["label", "path"])
    return new_df


if __name__ == '__main__':
    data = DataHolder(None, None, update=False)
    x_train, y_train, x_test, y_test = data.give_me_my_data()
    x_train = yolo_optimze_add_c(pd.concat([x_train, pd.DataFrame(y_train,columns=["label"])], axis=1))
