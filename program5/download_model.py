# Helper for PROGRAM 5: downloads the pretrained SSD MobileNet v3 (COCO) model.
# Run once from inside this folder:  uv run download_model.py

import tarfile
import urllib.request
from pathlib import Path

MODEL_URL = ("http://download.tensorflow.org/models/object_detection/"
             "ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz")
CONFIG_URL = ("https://raw.githubusercontent.com/ankityddv/ObjectDetector-OpenCV/"
              "main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

here = Path(__file__).parent
archive = here / "model.tar.gz"

if not (here / "frozen_inference_graph.pb").exists():
    print("Downloading weights (~20 MB) ...")
    urllib.request.urlretrieve(MODEL_URL, archive)

    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.endswith("frozen_inference_graph.pb"):
                member.name = "frozen_inference_graph.pb"
                tar.extract(member, here)

    archive.unlink()
    print("Saved frozen_inference_graph.pb")

if not (here / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt").exists():
    print("Downloading config ...")
    urllib.request.urlretrieve(CONFIG_URL, here / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    print("Saved ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

print("Done. Now add vehicle.jpg and run:  uv run main.py")
