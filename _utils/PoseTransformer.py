import torch
import json
from pathlib import Path
import os
import numpy as np
    
def alphaose2kineticsFormat(poses):
    skeleton = []
    for pose in poses:
        kp = torch.cat([x for x in pose['keypoints']], dim=0).tolist()
        skeleton.append({
            'pose': kp,
            'score': [x[0] for x in pose['kp_score'].tolist()]
        })
    return skeleton
    

def writeJson(dict, jsonPath : Path, filename):
    if not jsonPath.exists(): jsonPath.mkdir(parents=True)
    with (jsonPath / filename).open('w') as f:
        json.dump(dict, f)


def readJson(jsonFilename : Path):
    if os.path.getsize(jsonFilename) :
        with jsonFilename.open('r') as f:
            dict = json.load(f)
    else:
        dict = {}
    return dict

def coco2017Keypoints2CocoCut(coco2017, inputSize=[17, 3]):
    coco2017 = torch.FloatTensor(coco2017)
    res = torch.zeros(14, *(inputSize[1::]))
    res[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] = \
        coco2017[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    res[13] = (coco2017[5] + coco2017[6])/2.0
    return res.numpy()

# coco2017format skeleton to openposeCocoFormat skeleton
def coco2017Keypoints2openposeCoco(coco2017, inputSize=[17, 3]):
    # refer to : 
    # https://github.com/jin-s13/COCO-WholeBody, 
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#body-keypoint-ordering-in-c-python
    coco2017 = torch.FloatTensor(coco2017)
    res = torch.zeros(18, *(inputSize[1::]))
    res[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] = \
        coco2017[[0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
    res[1] = (coco2017[5] + coco2017[6])/2.0
    return res.numpy()

# coco2017format skeleton to openposeCocoFormat skeleton
def halpe26_2_haplpe26(coco2017, inputSize=[17, 3]):
    return np.array(coco2017)