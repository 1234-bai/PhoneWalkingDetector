import numpy as np
import cv2
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, select_device, save_one_box, loadData
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer as ADt
from utils.PointsUtils import getExtendenBox, pointsAnyInBox, xywh2xyxy
from utils.PoseTransformer import getBodyPartIndex


device=select_device(0)

poseTest = SingleImagePoseEstimation(
    checkpoint='weights/alphapose/fast_res50_256x192.pth',
    device=device
)

phoneTest= TargetsDetector(
    weights='weights/yolov5/phone_ep20.pt',
    device=select_device(0)
)

test =  TargetsDetector(
    weights='weights/yolov5/yolov5s.pt',
    device=device
)


extension = 0.4
dataset = loadData(source='datasets/testdata/images')
for path, im0s, vid_cap, s in dataset:
    # if dataset.mode == 'image': continue
    # 获得文件名字
    filename = path[0] if dataset.mode == 'stream' else path

    # 获得原始图片
    im0 = im0s[0] if dataset.mode == 'stream' else im0s # HWC, BGR

    # 检测人像
    _, peopleXyxyBoxes, confs, _ = test.detectSingleImage(im0, classes=[0], conf_thres=0.4)
    if(len(peopleXyxyBoxes) > 0):
        
        # 注释器（画图器）
        annotator = TargetsAnnotator(im0, 2)

        # 根据人像检测骨骼结点
        poses, _ = poseTest.process(im0, peopleXyxyBoxes, confs) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number

        for i,pose in enumerate(poses): #   对于每个有姿态的人像人像
            peopleBox = xywh2xyxy(pose['bbox'])
            annotator.box_label(peopleBox, label='people', color=colors(0))
            # keypoints = pose['keypoints']   # (W, H)
            wristpoints = pose['keypoints'][getBodyPartIndex('Mscoco', 'wrist')]
            crop = save_one_box(peopleBox, im0, BGR=True, save=False)
            _, phoneBoxes, _, _ = phoneTest.detectSingleImage(crop, classes=[0], conf_thres=0.4)
            for phoneBox in phoneBoxes:
                phoneBox += np.array(peopleBox)[[0, 1, 0, 1]]
                label = 'true' if pointsAnyInBox(wristpoints, phoneBox, extension) else 'false'
                annotator.box_label(phoneBox, label=label, color=colors(5), txt_color=colors(2))
                extenBox = getExtendenBox(phoneBox, ex=extension)
                annotator.box_label(extenBox, label='extension', color=colors(7))

        im0 = annotator.result()
        im0 = ADt.viewpPoseInImage(im0, poses, poseTest.getVisThres(), tracking=True)

    cv2.imshow(str(path), im0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


