import numpy as np
import cv2
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, select_device, LOGGER, loadData

phoneTest= TargetsDetector(
    weights='weights/yolov5/phoneEx5.pt',
    device=select_device(0)
)

dataset = loadData(source="datasets\stgcnTrainData\Mscoco\Stand")
# dataset = loadData(source="D:/QianXiaoYi/Pictures/Data/phone/make_by_myself/VID_20230409_165938.mp4")

totalTime = 0.0
capCount = 0
for path, im0s, vid_cap, s in dataset:
    capCount += 1
    im0 = im0s[0] if dataset.mode == "stream" else im0s
    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)
    cls, phoneXyxyBoxes, confs, time = phoneTest.detectSingleImage(im0, conf_thres=0.6)
    for i, box in enumerate(phoneXyxyBoxes):
        annotator.box_label(box, phoneTest.getLabelName(cls[i])+str(round(float(confs[i]), 2)), colors(2)) 
    img = annotator.result()
    LOGGER.info(f'{s} {time * 1E3:.1f}ms')
    totalTime += time
    cv2.imshow(str(path), img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
LOGGER.info(f'average time:{totalTime / capCount * 1E3:.1f}ms')