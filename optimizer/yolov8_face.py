# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from datamanagement.data_holder import DataHolder
from tqdm import tqdm

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
    x_train, y_train, x_test, y_test = data.give_me_my_data()
    faces = []
    for i in tqdm(x_train["path"].tolist()):
        faces.append(find_face(i))
    print(faces)

