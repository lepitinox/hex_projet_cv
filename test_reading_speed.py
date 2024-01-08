import timeit
import cv2
from PIL import Image
from pathlib import Path
paht = str(Path(r"D:\pycharm\hex_projet_cv\data\train\ade20k\images\labelme_jxyafpmktdyqaiq.jpg"))
print(timeit.timeit('Image.open(paht)', number=10, globals=globals()))
print(timeit.timeit('cv2.imread(paht)', number=10, globals=globals()))
