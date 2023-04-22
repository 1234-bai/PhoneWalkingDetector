import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5 import select_device, loadData, save_one_box
from libs.yolov5.yolov5DetectorApi import TargetsDetector
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation as PoseEstimation
from utils.PointsUtils import pointInBox, pointsAnyInBox, xywh2xyxy
from utils.PoseTransformer import getBodyPartIndex

def run(
    source,
    device,
    class_names,
    save_dir
):
    device = select_device(device)
    peopleDe = TargetsDetector('weights/yolov5/yolov5s.pt', device)
    phoneDe = TargetsDetector('weights/yolov5/phoneEx5.pt', device)
    pEs = PoseEstimation(device)
    save_dir = Path(save_dir)
    for cl in class_names:
        src = str(Path(source) / cl)
        dataset = loadData(src)
        wristExs = []
        earExs = []
        for path, im0, _, _ in tqdm(dataset, desc=src):
            img = im0[0] if dataset.mode == 'strem' else im0
            _, peopleBoxes, peoConfs, time = peopleDe.detectSingleImage(img, classes=[0])
            if len(peopleBoxes):
                poses = pEs.process(img, peopleBoxes, peoConfs)[0]
                for pose in poses:
                    peopleBox = xywh2xyxy(pose['bbox'])  # 获得此人的人像盒子xyxy
                    keypoints = pose['keypoints'] # 获得此人的骨骼结点
                    wristpoints = keypoints[getBodyPartIndex('Mscoco', 'wrist')]
                    earpoints = keypoints[getBodyPartIndex('Mscoco', 'ear')]
                    crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                    phoneBoxes = phoneDe.detectSingleImage(crop, conf_thres=0.6, classes=[0])[1]
                    if len(phoneBoxes):
                        phoneBoxes += np.array(peopleBox)[[0, 1, 0, 1]]
                        for phoneBox in phoneBoxes:
                            extension = 0.0
                            w = False
                            e = False
                            while(True):
                                if not w and pointsAnyInBox(wristpoints, phoneBox, extension):
                                    wristExs.append(extension)
                                    w = True
                                if not e and pointsAnyInBox(earpoints, phoneBox, extension):
                                    earExs.append(extension)
                                    e = True
                                if w and e: break
                                extension += 0.01
        with(save_dir / (cl+'.txt')).open('w') as f:
            f.write('wrist-extension:\n')
            f.write(' '.join(map(str, wristExs))+'\n')
            f.write('ear-extension:\n')
            f.write(' '.join(map(str, earExs))+'\n')




if __name__ == '__main__':
    run(
        source='datasets/stgcnTrainData/Mscoco/',
        device=0,
        class_names=['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'Stand', 'Sit', 'Other'],
        save_dir='runs/phone'
    )