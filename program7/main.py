# PROGRAM 7: Pedestrian Detection System
# Detects pedestrians using OpenCV and the Histogram of Oriented Gradients (HOG) descriptor.

import cv2
from pathlib import Path

folder = Path(__file__).parent
image = cv2.imread(str(folder / "pedestrians.jpg"))
if image is None:
    raise FileNotFoundError("pedestrians.jpg not found in the program7 folder")

# Resize so the detector runs at a sensible scale
image = cv2.resize(image, (640, int(640 * image.shape[0] / image.shape[1])))
output = image.copy()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

boxes, weights = hog.detectMultiScale(image, winStride=(4, 4),
                                      padding=(8, 8), scale=1.03)

min_score = 0.25
count = 0

for (x, y, w, h), score in zip(boxes, weights):

    if score < min_score:
        continue

    count += 1
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output, f"Person {float(score):.2f}", (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

print("Pedestrians detected:", count)

cv2.imwrite(str(folder / "output.jpg"), output)
