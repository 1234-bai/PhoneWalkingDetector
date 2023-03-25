import numpy as np
import cv2
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator, select_device
from libs.yolov5.utils.plots import colors
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer as ADt
from libs.st_gcn.TwoStreamStgcn import ActionEstimation
from _utils.PoseTransformer import coco2017Keypoints2CocoCut as Dt, toBoneboxCoord
from _utils.PointsUtils import kepoints2bbox, xywh2xyxy


device=select_device(0)
ae = ActionEstimation(
    # weight_file='libs/st_gcn/model/stgcn_class6_150_p90.pt',
    # class_names= ['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'photo', 'Stand', 'other'],
    # layout='Mscoco',
    device=device
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

preFilename = ''
for path, _, im0s, vid_cap, s in dataset:
    # if dataset.mode == 'image': continue
    # 获得文件名字
    filename = path[0] if dataset.mode == 'stream' else path

    # 获得原始图片
    im0 = im0s[0] if dataset.mode == 'stream' else im0s

    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)

    # 检测人像
    _, peopleXyxyBoxes, crops, confs = test.detectorSingleImg(im0, classes=[0], conf_thres=0.4)
    if(len(peopleXyxyBoxes) > 0):

        if dataset.mode == 'image':

            # 根据人像检测骨骼结点
            poses = poseTest.process(im0, peopleXyxyBoxes, confs) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number

            for i,pose in enumerate(poses): #   对于每个人像
                keypoints = pose['keypoints'] # 获得此人的骨骼结点
                scores = pose['kp_score'] # 获得此人的骨骼结点置信度
                    
                # 动作检测
                # 骨骼结点格式转换，与动作检测模型的骨骼结点输入格式匹配
                kp = Dt(keypoints, [17, 2]) 
                sc = Dt(scores, [17, 1])
                boneBox = kepoints2bbox(kp)   # 获得骨架盒子，注意和人体盒子相区分。
                kp -= boneBox[:2] # 获得相对于自身骨架盒子的坐标
                actionName = ae.predictSingleCap(kp, sc, boneBox[2:]-boneBox[:2])
                    
                annotator.box_label(peopleXyxyBoxes[i], label=ae.getLabel(actionName), color=colors(0))

        else:
            if preFilename != filename: # 新的视频或者第一个视频
                preFilename = filename
                # 旧的Tracker在tracker不指向它的时候，被Python垃圾回收机制自动回收
                poseTest.initTracker()
                poseStore = []  # 记录的之前帧存在的人员的记录
            # 根据人像检测骨骼结点
            poses = poseTest.process(im0, peopleXyxyBoxes, confs, tracking=True)
            existedPeo = ([], {}) # 当前帧上现存的人
            for ps in poses:
                id = ps['idx']
                existedPeo[0].append(id)
                existedPeo[1][id] = xywh2xyxy(ps['bbox'])
                kp = toBoneboxCoord(Dt(ps['keypoints'], [17, 2]), norm=True)
                sc = Dt(ps['kp_score'], [17, 1])
                vc = np.concatenate((kp,sc), axis=1)
                try:
                    poseStore[id].append(vc)
                except IndexError:
                    poseStore.append([vc])
            for id,tvc in enumerate(poseStore):
                if id in existedPeo[0]:
                    actionName = 'pending'
                    if len(tvc) >= 5:
                        actionIndex = ae.predict(np.array(tvc), None, True)
                        actionName = ae.getLabel(actionIndex)
                    annotator.box_label(existedPeo[1][id], label=str(id)+actionName, color=colors(0))
                else:
                    tvc.clear()


    img = annotator.result()
    img = ADt.viewpPoseInImage(img, poses, poseTest.getVisThres(), tracking=True)
    cv2.imshow(str(path), img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


