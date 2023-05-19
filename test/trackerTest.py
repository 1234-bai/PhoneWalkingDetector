import cv2
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.PointsUtils import xywh2xyxy
from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, loadData, select_device
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer as ADt


device=select_device(0)

poseTest = SingleImagePoseEstimation(
    device=device
)

test =  TargetsDetector(
    weights='weights/yolov5/yolov5s.pt',
    device=device
)
dataset = loadData(source='datasets/testdata/video/')


for path, im0s, vid_cap, s in dataset:

    # if dataset.mode == 'image': continue

    # 获得文件名字
    filename = path[0] if dataset.mode == 'stream' else path

    # 获得原始图片
    im0 = im0s[0] if dataset.mode == 'stream' else im0s
    im = im0.copy()

    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)
    annotator2 = TargetsAnnotator(im, 2)

    # 检测人像
    _, peopleXyxyBoxes, confs, _ = test.detectSingleImage(im0, classes=[0], conf_thres=0.4)
    if(len(peopleXyxyBoxes) > 0):

        # 根据人像检测骨骼结点
        poses,_ = poseTest.process(im0, peopleXyxyBoxes, confs, tracking=False) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number
            
        for i,pose in enumerate(poses): #   对于每个人像
            box = pose['bbox'] # xywh
            id = pose['idx']
            # annotator.box_label(xywh2xyxy(box), label='people:'+str(round(float(pose['proposal_score']),2)), color=colors(0))
            annotator.box_label(xywh2xyxy(box), label=str(i)+','+str(id), color=colors(0))
            annotator2.box_label(peopleXyxyBoxes[i], label=str(i), color=colors(0))

    img = annotator.result()
    img = ADt.viewpPoseInImage(img, poses, poseTest.vis_thres, tracking=False)
    cv2.imshow('alpha', img)
    img = annotator2.result()
    cv2.imshow('yolov5', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


