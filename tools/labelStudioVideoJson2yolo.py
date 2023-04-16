from pathlib import Path
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
import argparse

def readJson(jsonFilename : Path):
    if jsonFilename.exists() and os.path.getsize(jsonFilename) :
        with jsonFilename.open('r') as f:
            dict = json.load(f)
    else:
        dict = {}
    return dict


def xywh2centerwh(box):
    box = np.array(box)
    box[:2] += (box[2:] / 2) # centerX, centerY, w, h
    return box.tolist()


def xformat(num):
    return str(round(num, 6))

pointers = None

def parseBoxes(boxesDict, labelsDict):
    boxKeyFrames = []
    labels = []
    for box in boxesDict:
        boxKf = []
        frames = box['sequence']
        for kf in frames:   # keyframe
            fi = kf['frame'] # frame id
            x, y, width, height = kf['x'], kf['y'], kf['width'], kf['height'] # x% , y%, width%, height%
            boxKf.append([fi, x/100, y/100, width/100, height/100])
        boxKeyFrames.append(boxKf)
        labels.append(labelsDict[box['labels'][0]])
    global pointers
    pointers = [0] * len(boxKeyFrames)
    return (boxKeyFrames, labels)


def getCoordsBetweenTwoKeyFrames(lastKf, nextKf, frameId):
    lastKf = np.array(lastKf)
    nextKf = np.array(nextKf)
    delt = nextKf - lastKf
    now = lastKf + (frameId - lastKf[0]) / delt[0] * delt
    assert(round(now[0]) == frameId)
    return xywh2centerwh(now[1:])

def getFrameBoxes(boxes, labels, frameId):
    res = []
    global pointers # index in box of next key frame which have't visited
    for i, box in enumerate(boxes):
        pointer = pointers[i]
        if pointer == -1:
            continue
        if frameId < box[0][0]: # frame id
            continue
        nextKf = box[pointer]   # keyframe
        if frameId == nextKf[0]: # frame id
            res.append([
                labels[i], *(xywh2centerwh(nextKf[1:]))
            ])
            pointer += 1
            if pointer == len(box):
                pointer = -1
            pointers[i] = pointer 
            continue
        if frameId < nextKf[0]:
            lastKf = box[pointer - 1] 
            res.append([
                labels[i], *(getCoordsBetweenTwoKeyFrames(lastKf, nextKf, frameId))
            ])
            continue
    return res

def run(
    input_json,
    input_videos_dir,
    output_dir,
    labelNames,
):
    opd = Path(output_dir)
    (opd / 'images').mkdir(parents=True, exist_ok=True)
    (opd / 'labels').mkdir(parents=True, exist_ok=True)
    ivd = Path(input_videos_dir)
    labelDic = {label : i for i, label in enumerate(labelNames)}
    dic = readJson(Path(input_json))
    sumTxtPath = Path(opd / 'test.txt')
    f = sumTxtPath.open("w"); f.close()
    for vf in dic:  # video file
        videoname = Path(vf['video'])
        videostem = videoname.stem
        videoname = videoname.name
        cap = cv2.VideoCapture(str(ivd / videoname))
        boxes = vf['box']
        boxKeyFrames, classes = parseBoxes(boxes, labelDic) 
        framesCount = boxes[0]['framesCount']
        for i in tqdm(range(framesCount), desc=videoname):
            _, frame = cap.read()
            imgName = videostem + f'_{i+1}.jpg'
            cv2.imwrite(str(opd / 'images' / imgName), frame) #存入快照
            clses = getFrameBoxes(boxKeyFrames, classes, i+1)
            txtPath = Path(opd / 'labels' / (videostem + f'_{i+1}.txt'))
            with txtPath.open("w") as f:
                for cls in clses:
                    f.write(' '.join(map(xformat, cls))+'\n')
            with sumTxtPath.open("a") as f:
                f.write('./images/'+imgName+'\n')
        cap.release()


if __name__ == "__main__":
    run(
        input_json='datasets/yolodata/phonewalking_test.json',
        input_videos_dir='C:/Users/QianXiaoYi/AppData/Local/label-studio/label-studio/media/upload/1',
        output_dir='datasets/yolodata/test',
        labelNames=['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'Walking', 'Other']
    )


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', type=str, default='datasets/testdata/images', help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--nosave', action='store_true', help='save images/videos result')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--save-dir', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
#     opt = parser.parse_args()
#     return opt


# def main(opt):
#     run(**vars(opt))


# if __name__ == '__main__':
#     opt = parse_opt()
#     main(opt)