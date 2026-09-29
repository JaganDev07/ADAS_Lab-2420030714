# PROGRAM 5: Vehicle Detection System
# Detects vehicles in a road image and draws bounding boxes around them.
# Uses the OpenCV DNN module with a pretrained SSD MobileNet v3 (COCO) model.
# Model files needed in this folder (see README):
#   frozen_inference_graph.pb
#   ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

import os
import cv2
import numpy as np
from pathlib import Path

folder = Path(__file__).parent
CONFIG = folder / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS = folder / "frozen_inference_graph.pb"

for f in (CONFIG, WEIGHTS):
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found - download the SSD MobileNet model (see README)")

classNames = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window",
    "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

net = cv2.dnn_DetectionModel(WEIGHTS, CONFIG)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

image = cv2.imread(str(folder / "vehicle.jpg"))
if image is None:
    raise FileNotFoundError("vehicle.jpg not found in the program5 folder")

output = image.copy()

classIds, scores, boxes = net.detect(image, confThreshold=0.45, nmsThreshold=0.40)

classIds = np.array(classIds).flatten()
scores = np.array(scores).flatten()

vehicle_names = {"car", "motorcycle", "bus", "truck", "bicycle"}
count = 0

for classId, score, box in zip(classIds, scores, boxes):

    classId = int(classId)
    if classId >= len(classNames):
        continue

    name = classNames[classId]
    if name not in vehicle_names:
        continue

    x, y, w, h = box

    # Ignore boxes covering most of the frame - they are false positives
    if w * h > 0.4 * image.shape[0] * image.shape[1]:
        continue

    count += 1
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output, f"{name} {score:.2f}", (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

print("Vehicles detected:", count)

cv2.imwrite(str(folder / "output.jpg"), output)
