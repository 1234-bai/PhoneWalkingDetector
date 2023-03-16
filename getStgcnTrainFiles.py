import numpy as np
import cv2
from pathlib import Path
import random
import os

from libs.yolov5.yolov5DetectorApi import TargetsDecetor
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from _utils.PoseTransformer import writeJson, readJson, alphaose2kineticsFormat

def jsonPosePack(poses, label_index, label_name):
    data = [{
        "frame_index" : 0,
        "skeleton" : alphaose2kineticsFormat(poses)
    }] if poses is not None and len(poses) > 0 else []
    return {
        "data" : data,
        "label" : label_name,
        "label_index" : label_index 
    }

def writePoseJson(poses, label_index, label_name, jsonPath : Path, filename):
    poseDict = jsonPosePack(poses, label_index, label_name)
    writeJson(poseDict, jsonPath, filename)


name = 'halpe26'
input_dir = Path("D:/QianXiaoYi/Pictures/Data/train_with_anoations/0_phone/images")
output_dir = Path('stgcnTrainData/')
output_train_json_dir = output_dir / (f'{name}_train')
output_val_json_dir = output_dir / (f'{name}_val')
output_train_json = f'{name}_train_label.json'
output_val_json = f'{name}_val_label.json'

valThres = 8
outTrainSumJson = readJson(output_dir / output_train_json)
outValSumJson = readJson(output_dir / output_val_json)
label_names = ['Other', 'Play', 'Call']
copy_dir = [(output_dir / x) for x in label_names]
for x in copy_dir:
    x.mkdir(parents=True, exist_ok=True)

peopleDec =  TargetsDecetor(
    weights='D:/_NewCode/PythonPro/Phone_Walking_Detector/libs/yolov5/weights/yolov5s.pt',
    data='libs/yolov5/data/coco128.yaml',
    device=0
)

poseEst = SingleImagePoseEstimation(
    configFilePath='libs/Alphapose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
    checkpoint='libs/Alphapose/pretrained_models/halpe26_fast_res50_256x192.pth',
    device=0
)

dataset = peopleDec.loadData(source=input_dir)

for j, (path, _, im0s, vid_cap, s) in enumerate(dataset):

    im0 = im0s
    _, peoXyxyBoxes, _, confs= peopleDec.detectorSingleImg(im0, classes=[0], conf_thres=0.45)
    file = Path(path)
    if(len(peoXyxyBoxes) > 0) :
        poses = poseEst.process(im0, peoXyxyBoxes, confs, normalizelCrood=True)
        filename = file.stem
        cv2.imshow(filename, im0)
        choice =  cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if choice == ord('b'): break
        for i, label in enumerate(label_names):
            if choice == ord(str(i)):
                imgDict = {
                    "has_skeleton": poses is not None and len(poses) > 0, 
                    "label": label, 
                    "label_index": i
                }
                if(j % 10 < valThres):    # train set
                    writePoseJson(poses, i, label, output_train_json_dir, (filename+'.json'))
                    outTrainSumJson[filename] = imgDict
                    print('train')
                else:   #   val set, last 400 images
                    writePoseJson(poses, i, label, output_val_json_dir, (filename+'.json'))
                    outValSumJson[filename] = imgDict
                    print('val')
                assert(cv2.imwrite(copy_dir[i] / file.name, im0))
                # file.unlink()
                break
    # else:
    #     file.unlink()
writeJson(outTrainSumJson, output_dir, output_train_json)
writeJson(outValSumJson, output_dir, output_val_json) 

# 将分完的图片删除掉
# 删除目标文件夹里的隐藏文件