import numpy as np
import cv2

from libs.yolov5.yolov5DetectorApi import TargetsDecetor, TargetsAnnotator
from libs.yolov5.utils.plots import colors
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer
from libs.st_gcn.StgcnApi import ActionEstimation

from MathUtils import twoPointsSuperpose, getBoxCenters


ae = ActionEstimation(weight_file='libs\st_gcn\model\st-gcn-tsstg-fail-model.pth')

poseTest = SingleImagePoseEstimation(
    configFilePath='libs\\Alphapose\\configs\\coco\\resnet\\256x192_res50_lr1e-3_1x.yaml',
    checkpoint='libs\\Alphapose\\pretrained_models\\fast_res50_256x192.pth',
    device=0
)

test =  TargetsDecetor(
    weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\yolov5s.pt',
    data='libs\\yolov5\\data\\coco128.yaml'
)
dataset = test.loadData(source='images')

phoneTest= TargetsDecetor(
    weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\phone_ep20.pt',
    data='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\data\phone.yaml'
)


for path, _, im0s, vid_cap, s in dataset:

    # 获得原始图片
    im0 = im0s[0] if dataset.mode == 'stream' else im0s

    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)

    # 检测人像
    _, peopleXyxyBoxes, crops, confs = test.detectorSingleImg(im0, classes=[0])
    if(len(peopleXyxyBoxes) > 0):

        # 根据人像检测骨骼结点
        poses = poseTest.process(path, im0, peopleXyxyBoxes, confs) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number

        # 对每个人像进行手机检测和动作检测
        handsIndex = poseTest.getHandIndex() # 获得该骨骼结点格式的手部结点编号
        earsIndex = poseTest.getEarIndex()  # 获得该骨骼结点格式的耳朵结点编号
        for i,crop in enumerate(crops): #   对于每个人像
            keypoints = poses[i]['keypoints'] # 获得此人的骨骼结点
            handCenters = [keypoints[handsIndex[0]], keypoints[handsIndex[1]]] # 获得手部结点
            earCenters = [keypoints[earsIndex[0]], keypoints[earsIndex[1]]]
            scores = poses[i]['kp_score'] # 获得此人的骨骼结点置信度
            
            # 动作检测
            # 骨骼结点格式转换，与动作检测模型的骨骼结点输入格式匹配
            keypoints = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(keypoints, [17, 2])
            scores = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(scores, [17, 1])
            box = poses[i]['bbox'] # 每个人像的xywhBox
            actionName = ae.predict(keypoints, scores, (box[2], box[3]))
            if(actionName != 'Walking'):
                continue

            # 手机检测
            _, phoneXyxyBoxes,_, _ = phoneTest.detectorSingleImg(crop, classes=[0])
            if(len(phoneXyxyBoxes)):   # 人像图中存在手机
                phoneCenters = getBoxCenters(phoneXyxyBoxes)
                for j, pc in enumerate(phoneCenters):  # 对于每个手机，是否与人手重合
                    for hc in handCenters:
                        if twoPointsSuperpose(pc, hc, crop.shape): # 重合则开始验证动作
                            annotator.box_label(peopleXyxyBoxes[i], label=actionName, color=colors(0))
                            annotator.double_box_label(peopleXyxyBoxes[i], phoneXyxyBoxes[j], label='phone', color=colors(5))

    img = annotator.result()
    cv2.imshow(str(path), img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


