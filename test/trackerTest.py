import numpy as np
import cv2
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from _utils.PointsUtils import kepoints2bbox, xywh2xyxy
from _utils.PoseTransformer import nochange as Dt
from libs.st_gcn.StgcnApi import ActionEstimation
from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator, select_device
from libs.yolov5.utils.plots import colors
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer as ADt


device=select_device(0)
ae = ActionEstimation(
    weight_file='libs/st_gcn/model/stgcn_class6_150_p90.pt',
    class_names= ['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'photo', 'Stand', 'other'],
    layout='Mscoco'
)

poseTest = SingleImagePoseEstimation(
    configFilePath='libs/Alphapose/configs/coco_256x192_res50_lr1e-3_1x.yaml',
    checkpoint='libs/Alphapose/pretrained_models/fast_res50_256x192.pth',
    device=device
)

# poseTest = SingleImagePoseEstimation(
#     configFilePath='libs/Alphapose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
#     checkpoint='libs/Alphapose/pretrained_models/halpe26_fast_res50_256x192.pth',
#     device=0
# )

test =  TargetsDetector(
    weights='D:/_NewCode/PythonPro/Phone_Walking_Detector/libs/yolov5/weights/yolov5s.pt',
    data='libs/yolov5/data/coco128.yaml',
    device=device
)
dataset = test.loadData(source=0)


for path, _, im0s, vid_cap, s in dataset:

    if dataset.mode == 'image': continue

    # 获得文件名字
    filename = path[0] if dataset.mode == 'stream' else path

    # 获得原始图片
    im0 = im0s[0] if dataset.mode == 'stream' else im0s

    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)

    # 检测人像
    _, peopleXyxyBoxes, confs = test.detectorSingleImg(im0, classes=[0], conf_thres=0.4)
    if(len(peopleXyxyBoxes) > 0):

        # 根据人像检测骨骼结点
        poses = poseTest.process(im0, peopleXyxyBoxes, confs, tracking=True) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number

        for i,pose in enumerate(poses): #   对于每个人像
            box = pose['bbox'] # xywh
            id = pose['idx']
            annotator.box_label(xywh2xyxy(box), label=str(id), color=colors(2*i))

    img = annotator.result()
    img = ADt.viewpPoseInImage(img, poses, poseTest.getVisThres(), tracking=True)
    cv2.imshow(str(path), img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


