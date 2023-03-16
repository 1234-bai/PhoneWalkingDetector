import numpy as np
import cv2
from pathlib import Path
import random
import os

from libs.yolov5.yolov5DetectorApi import TargetsDecetor
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from _utils.PoseTransfromer import PoseDataTransformer as PT

def jsonPosePack(poses, label_index, label_name):
    data = [{
        "frame_index" : 1,
        "skeleton" : PT.alphaose2kineticsFormat(poses)
    }] if poses is not None and len(poses) > 0 else []
    return {
        "data" : data,
        "label" : label_name,
        "label_index" : label_index 
    }

def writePoseJson(poses, label_index, label_name, jsonPath : Path, filename):
    poseDict = jsonPosePack(poses, label_index, label_name)
    PT.writeJson(poseDict, jsonPath, filename)


input_dir = Path('D:\\QianXiaoYi\\Pictures\\Data\\train_with_anoations\\0_phone\\images')
output_dir = Path('stgcnTrainData/')
output_train_json_dir = output_dir / 'kinetics_train'
output_val_json_dir = output_dir / 'kinetics_val'
output_train_json = 'kinetics_train_label.json'
output_val_json = 'kinetics_val_label.json'

trainThres = int(0.8 * len(os.listdir(input_dir)))
print(trainThres)
outTrainSumJson = PT.readJson(output_dir / output_train_json)
outValSumJson = PT.readJson(output_dir / output_val_json)
label_names = ['Call', 'Play', 'Hold', 'other']

peopleDec =  TargetsDecetor(
    weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\yolov5s.pt',
    data='libs\\yolov5\\data\\coco128.yaml',
    device=0
)

poseEst = SingleImagePoseEstimation(
    configFilePath='libs\\Alphapose\\configs\\halpe_26\\resnet\\256x192_res50_lr1e-3_1x.yaml',
    checkpoint='libs\\Alphapose\\pretrained_models\\alphapose-halpe26_fast_res50_256x192.pth',
    device=0
)

dataset = peopleDec.loadData(source=input_dir)

for i, (path, _, im0s, vid_cap, s) in enumerate(dataset):
    if i == 20: break
    im0 = im0s
    _, peoXyxyBoxes, _, confs= peopleDec.detectorSingleImg(im0, classes=[0])
    if(len(peoXyxyBoxes) > 0) :
        poses = poseEst.process(im0, peoXyxyBoxes, confs, normalizelCrood=True)
        filename = Path(path).stem
        cv2.imshow(filename, im0)
        choice =  cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if choice == ord('b') : break
        for i, label in enumerate(label_names):
            if choice == ord(str(i)):
                imgDict = {
                    "has_skeleton": poses is not None and len(poses) > 0, 
                    "label": label, 
                    "label_index": i
                }
                if(i > trainThres):    # train set
                    writePoseJson(poses, i, label, output_train_json_dir, (filename+'.json'))
                    outTrainSumJson[filename] = imgDict
                else:   #   val set
                    writePoseJson(poses, i, label, output_val_json_dir, (filename+'.json'))
                    outValSumJson[filename] = imgDict
                break
        # Path(path).unlink()
PT.writeJson(outTrainSumJson, output_dir, output_train_json)
PT.writeJson(outValSumJson, output_dir, output_val_json, ) 

# 固定随机数
# 将分完的图片删除掉