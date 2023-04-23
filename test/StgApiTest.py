import numpy as np
import cv2
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, LOGGER, select_device, loadData
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer as ADt
from libs.st_gcn.TwoStreamStgcn import ActionEstimation as StandAe
from libs.st_gcn.StgcnApi import ActionEstimation as WalkAe
from utils.PoseTransformer import coco2017Keypoints2CocoCut, nochange, toBoneboxCoord
from utils.PointsUtils import xywh2xyxy


device=select_device(0)

model = 'walk'

ae = WalkAe(
    weight_file='weights/stgcn/stgcn_class3_150_94_ex9.pt',
    class_names= ['no', 'one', 'two'],
    layout='Mscoco',
    device=device
) if model == 'walk' else StandAe(device=device)

Dt = nochange if model == 'walk' else coco2017Keypoints2CocoCut

poseTest = SingleImagePoseEstimation(
    checkpoint='weights/alphapose/fast_res50_256x192.pth',
    device=device
)

test =  TargetsDetector(
    weights='weights/yolov5/yolov5s.pt',
    device=device
)
source='datasets/testdata/images/'
dataset = loadData(source=source)


preFilename = ''
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
    peTime = 0.0
    aeTime = 0.0

    # 检测人像
    _, peopleXyxyBoxes, confs, _ = test.detectSingleImage(im0, classes=[0], conf_thres=0.4)
    if(len(peopleXyxyBoxes) > 0):

        # 注释器（画图器）
        annotator = TargetsAnnotator(im0, 2)

        if dataset.mode == 'image':

            # 根据人像检测骨骼结点
            poses, time = poseTest.process(im0, peopleXyxyBoxes, confs) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number
            peTime += time

            for pose in poses: #   对于每个人像
                keypoints = pose['keypoints'] # 获得此人的骨骼结点
                scores = pose['kp_score'] # 获得此人的骨骼结点置信度
                peopleBox = xywh2xyxy(pose['bbox'])
                    
                # 动作检测
                # 骨骼结点格式转换，与动作检测模型的骨骼结点输入格式匹配
                kp = Dt(keypoints, [17, 2]) 
                sc = Dt(scores, [17, 1])
                # boneBox = kepoints2bbox(kp)   # 获得骨架盒子，注意和人体盒子相区分。
                kp = toBoneboxCoord(kp, norm=True) # 获得相对于自身骨架盒子的坐标
                actionName, time = ae.predictSingleCap(kp-0.5, sc, None, normed=True)
                aeTime += time
                    
                annotator.box_label(peopleBox, label=ae.getLabel(actionName), color=colors(0))

        else:

            if preFilename != filename: # 新的视频或者第一个视频
                preFilename = filename
                # 旧的Tracker在tracker不指向它的时候，被Python垃圾回收机制自动回收
                poseTest.initTracker()
                poseStore = []  # 记录的之前帧存在的人员的记录

            # 根据人像检测骨骼结点
            poses, time = poseTest.process(im0, peopleXyxyBoxes, confs, tracking=True)
            peTime += time

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
            for id,ntvc in enumerate(poseStore):
                if id in existedPeo[0]:
                    actionName = 'pending'
                    if len(ntvc) >= 5:
                        actionIndex, time = ae.predict(np.array(ntvc), None, True)
                        aeTime += time
                        actionName = ae.getLabel(actionIndex)
                    annotator.box_label(existedPeo[1][id], label=str(id)+actionName, color=colors(0))
                else:
                    ntvc.clear()

        
        im0 = annotator.result()
        im0 = ADt.viewpPoseInImage(im0, poses, poseTest.vis_thres, tracking=True)

    LOGGER.info(f"{s}\n      pose esatimation time: {peTime * 1E3:.1f}ms")
    LOGGER.info(f"      action esatimation time: {aeTime * 1E3:.1f}ms")
    totalAeTime += aeTime
    totalPeTime += peTime
    cv2.imshow(filename, im0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

LOGGER.info(f"{source}, average process time: {(totalPeTime + totalAeTime) / capCount * 1E3:.1f}ms")
