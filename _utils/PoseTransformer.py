import torch
import json
from pathlib import Path
import os
import numpy as np
from .PointsUtils import toBoneboxCoord


def alphaose2kineticsFormat(poses, norm=False):
    skeleton = []
    for pose in poses:
        kp = toBoneboxCoord(pose['keypoints'], norm)
        kp = torch.cat([x for x in kp], dim=0).tolist()
        kp = [round(x, 3) for x in kp]
        skeleton.append({
            'pose': kp,
            'score': [round(x[0],3) for x in pose['kp_score'].tolist()]
        })
    return skeleton
    

def writeJson(dict, jsonPath : Path, filename):
    if not jsonPath.exists(): jsonPath.mkdir(parents=True)
    with (jsonPath / filename).open('w') as f:
        json.dump(dict, f)


def readJson(jsonFilename : Path):
    if jsonFilename.exists() and os.path.getsize(jsonFilename) :
        with jsonFilename.open('r') as f:
            dict = json.load(f)
    else:
        dict = {}
    return dict

def coco2017Keypoints2CocoCut(coco2017, inputSize=[17, 3]):
    typeflag = 0
    if isinstance(coco2017, np.ndarray):
        typeflag = 1
    elif isinstance(coco2017, torch.Tensor):
        typeflag = 2
    coco2017 = torch.FloatTensor(coco2017)
    res = torch.zeros(14, *(inputSize[1::]))
    res[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] = \
        coco2017[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    res[13] = (coco2017[5] + coco2017[6])/2.0
    return res if typeflag == 2 else (res.tolist() if typeflag == 0 else res.numpy())

# coco2017format skeleton to openposeCocoFormat skeleton
def coco2017Keypoints2openposeCoco(coco2017, inputSize=[17, 3]):
    # refer to : 
    # https://github.com/jin-s13/COCO-WholeBody, 
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#body-keypoint-ordering-in-c-python
    typeflag = 0
    if isinstance(coco2017, np.ndarray):
        typeflag = 1
    elif isinstance(coco2017, torch.Tensor):
        typeflag = 2
    coco2017 = torch.FloatTensor(coco2017)
    res = torch.zeros(18, *(inputSize[1::]))
    res[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] = \
        coco2017[[0, 5, 7, 9, 6, 8, 20, 11, 13, 15, 12, 14, 16, 1, 2, 3, 4]]
    res[1] = (coco2017[5] + coco2017[6])/2.0
    return res if typeflag == 2 else (res.tolist() if typeflag == 0 else res.numpy())


def coco2017Kps2coco2017cut(coco2017, inputSize=[17, 3]):
    return coco2017[:13]


def nochange(coco2017, inputSize=[17, 3]):
    return coco2017


def getBodyPartIndex(keypoinesType = 'coco2017', bodyPartType = 'wrist'):
    keypoinesType = keypoinesType.lower()
    bodyPartType = bodyPartType.lower()
    if bodyPartType == 'wrist':
        if keypoinesType == 'coco2017' or keypoinesType == 'mscoco':
            return [9, 10]
        elif keypoinesType == 'openpose25' or keypoinesType == 'openposecoco':
            return [4,7]
        elif keypoinesType == 'cococut':
            return [5, 6]
        elif keypoinesType == 'halpe_26':
            return [9, 10]
    elif bodyPartType == 'ear':
        if keypoinesType == 'coco2017' or keypoinesType == 'mscoco':
            return [3, 4]
        elif keypoinesType == 'openpose25':
            return [17, 18]
        elif keypoinesType == 'openposecoco':
            return [16,17]
        elif keypoinesType == 'cococut':
            return None
        elif keypoinesType == 'halpe_26':
            return [3, 4]
    else:
        raise
        return None
