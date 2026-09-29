# PROGRAM 1: Automatic Headlight Control System
# Turns the headlights ON or OFF based on the ambient light intensity.

import cv2
import numpy as np
from pathlib import Path

image = cv2.imread(str(Path(__file__).parent / "img-night.jpg"))
if image is None:
    raise FileNotFoundError("img-night.jpg not found in the program1 folder")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

average_brightness = np.mean(gray_image)

threshold = 85.0

print("Ambient brightness:", round(average_brightness, 2))

if average_brightness > threshold:
    print("DAY  -> Headlights OFF")
else:
    print("NIGHT -> Headlights ON")
