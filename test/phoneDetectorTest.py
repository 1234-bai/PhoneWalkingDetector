import numpy as np
import cv2
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, select_device, LOGGER

phoneTest= TargetsDetector(
    weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\phone_ep20.pt',
    data='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\data\phone.yaml',
    device=select_device(0)
)

dataset = phoneTest.loadData(source='data/images')

totalTime = 0.0
capCount = 0
for path, _, im0s, vid_cap, s in dataset:
    capCount += 1
    im0 = im0s[0] if dataset.mode == "stream" else im0s
    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)
    _, phoneXyxyBoxes, confs, time = phoneTest.detectorSingleImg(im0, classes=[0])
    for i, box in enumerate(phoneXyxyBoxes):
        annotator.box_label(box, str(round(float(confs[i]), 2)), colors(2))
    img = annotator.result()
    LOGGER.info(f'process time:{time * 1E3:.1f}ms')
    totalTime += time
    cv2.imshow(str(path), img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
LOGGER.info(f'average time:{totalTime / capCount * 1E3:.1f}ms')