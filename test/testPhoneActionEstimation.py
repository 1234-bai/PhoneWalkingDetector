import cv2
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5 import LOGGER, select_device, loadData
from detectors import YoloPhoneActionEstimation
from detectors.StgcnActionEstimation import PhoneActionEstimation

device=select_device(0)

model = 2

ae = PhoneActionEstimation(device, model) if isinstance(model, int) else YoloPhoneActionEstimation(device)

source='datasets/testData/images/'
dataset = loadData(source=source)


# time
totalPeTime = 0.0
totalAeTime = 0.0
capCount = 0

for path, im0s, vid_cap, s in dataset:
    # if dataset.mode == 'image': continue
    # 获得文件名字
    filename = path[0] if dataset.mode == 'stream' else path

    # 获得原始图片
    im0 = im0s[0] if dataset.mode == 'stream' else im0s

    # time
    capCount += 1

    cls, boxes, confs, crops, img, time = ae.detectSingleImage(im0=im0, conf_thres=0.5)

    peTime, aeTime = time[1], time[3]
    LOGGER.info(f"{s}\n      pose esatimation time: {peTime * 1E3:.1f}ms")
    LOGGER.info(f"      action esatimation time: {aeTime * 1E3:.1f}ms")
    totalAeTime += aeTime
    totalPeTime += peTime
    cv2.imshow(filename, img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

LOGGER.info(f"{source}, average process time: {(totalPeTime + totalAeTime) / capCount * 1E3:.1f}ms")
