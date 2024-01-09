# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from datamanagement.data_holder import DataHolder
from tqdm import tqdm
import cv2
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)


def find_face(img_path):
    imgs = Image.open(img_path)
    output = model(imgs)
    results = Detections.from_ultralytics(output[0])
    return results



if __name__ == '__main__':
    data = DataHolder(None, None, update=False)
    data.balanced_sample(1)
    x_train, y_train, x_test, y_test = data.give_me_my_data()
    for i in range(10):
        pth = x_train["path"].tolist()[i]
        base_img = cv2.imread(pth)
        faces = find_face(pth)
        for face in faces.xyxy:
            face = map(int, face)
            x, y, x2, y2 = face
            # draw a rectangle over the pixels
            cv2.rectangle(base_img, (x, y), (x2, y2), (0, 0, 255), 1)
        cv2.imshow('img', base_img)
        cv2.waitKey(0)



