import numpy as np
import cv2

from libs.yolov5.yolov5DetectorApi import TargetsDecetor, TargetsAnnotator
from libs.yolov5.utils.plots import colors

phoneTest= TargetsDecetor(
    weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\phone_ep20.pt',
    data='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\data\phone.yaml'
)

dataset = phoneTest.loadData(source='images')


for path, _, im0s, vid_cap, s in dataset:
    im0 = im0s
    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)
    _, phoneXyxyBoxes, _, confs= phoneTest.detectorSingleImg(im0, classes=[0])
    for i, box in enumerate(phoneXyxyBoxes):
        annotator.box_label(box, str(round(float(confs[i]), 2)))
    img = annotator.result()
    cv2.imshow(str(path), img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

