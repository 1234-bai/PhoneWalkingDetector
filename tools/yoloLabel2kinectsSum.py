from tqdm import tqdm
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5 import select_device, LoadImagesAndLabels
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from utils.PoseTransformer import writeJson, readJson, alphaose2kineticsFormat

device = select_device(0)
poseEst = SingleImagePoseEstimation(
    checkpoint='weights/alphapose/fast_res50_256x192.pth',
    device=device
)

def jsonPosePack(poses, label_index, label_name, copyTimes):
    data = []
    if poses is not None and len(poses) > 0:
        skeleton = alphaose2kineticsFormat(poses, True)
        data = [{
            "frame_index" : i,
            "skeleton" : skeleton
        } for i in range(copyTimes)]
    return {
        "data" : data,
        "label" : label_name,
        "label_index" : label_index 
    }

def writePoseJson(poses, label_index, label_name, copyTimes, jsonPath : Path, filename):
    poseDict = jsonPosePack(poses, label_index, label_name, copyTimes)
    writeJson(poseDict, jsonPath, filename)


def transform(
    name = 'Mscoco',
    input_root_dir = '',
    output_dir = '',
    label_names = ['Walk', 'oneHand', 'twoHands', 'other'],
    parts = ['val', 'train'],
    frameCount = 30
):
    output_dir = Path(output_dir)
    input_root_dir = Path(input_root_dir)
    for part in parts:
        jsons_dir = output_dir / (f'{name}_{part}')
        sum_json_name = f'{name}_{part}_label.json'
        sum_dict = readJson(output_dir / sum_json_name)
        path = str(input_root_dir / (part + '.txt'))
        dataset = LoadImagesAndLabels(path)
        for im0, labels, path in tqdm(dataset, desc=part):
            file = Path(path)
            filename = file.stem
            peoXyxyBoxes = labels[:, 1:]
            poses, _ = poseEst.process(im0.numpy(), peoXyxyBoxes, [0.8] * len(peoXyxyBoxes))
            if len(poses) == 1:
                pose = poses[0]
                label_index = int(labels[int(pose['idx'])][0])
                label_name = label_names[label_index]
                sum_dict[filename] = {
                    "has_skeleton": poses is not None and len(poses) > 0, 
                    "label": label_name, 
                    "label_index": label_index
                }
                writePoseJson(poses, label_index, label_name, frameCount, jsons_dir, (filename+'.json'))
        writeJson(sum_dict, output_dir, sum_json_name)


if __name__ == '__main__':
    transform(
        name='Mscoco',
        input_root_dir='datasets/actionData/yolo',
        output_dir='datasets/actionData/stgcn',
        label_names=['Walk', 'oneHand', 'twoHands', 'other'],
        parts = ['val', 'train'],
        frameCount = 30
    )