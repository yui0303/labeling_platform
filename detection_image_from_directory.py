#================================================================
#   File name   : detection_image_from_directory.py
#   reference   : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#================================================================

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

# set output directory
output_img_dir = "DETECT_IMAGES"
if not os.path.isdir(output_img_dir):
    os.mkdir(output_img_dir)

# image directory
img_dir = os.path.join(os.getcwd(), "prepare_data","without_annotation", "train")

img_list = [_ for _ in os.listdir(img_dir) if _.endswith(r".jpg")]

# load model
yolo = Load_Yolo_model()

# if len(sys.argv) > 1:
#     image = sys.argv[1]
#     if image + ".jpg" not in img_list:
#         raise NameError(image + ".jpg" + " is not in the directory")
#     elif image + ".jpg" in img_list:
#         path = os.path.join(img_dir, image + ".jpg")
#         output_path = os.path.join(output_img_dir, image + "_detect.jpg")
#         detect_image(yolo, path, output_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#         print('\033[91m', image + ".jpg", "detection finished.", '\033[0m')
# else:

XML = False
if len(sys.argv) > 1:
    XML = True

for img in img_list:
    path = os.path.join(img_dir, img)
    filename = os.path.splitext(img)[0]
    output_path = ''
    
    detect_image(yolo, path, output_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), XML = True, filename = filename)
    #print('\033[91m', img, "detection finished.", '\033[0m')
