import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector
from libs.yolov5 import select_device, loadData
from _utils.PointsUtils import xyxy2centerwh


input_dir = Path("datasets/stgcnTrainData/Mscoco")
output_dir = Path(f'datasets/stgcnTrainData/Mscoco/out')

label_names = ['Call', 'PlayWithOneHand', 'PlayWithTwoHands',  'Stand', 'Sit', 'Photograph','Other'] # 根据动作分类，而不是手机出现的位置
label_dir = [(output_dir / x) for x in label_names]
for x in label_dir:
    x.mkdir(parents=True, exist_ok=True)

device = select_device(0)
peopleDec =  TargetsDetector(
    weights='D:/_NewCode/PythonPro/Phone_Walking_Detector/libs/yolov5/weights/yolov5s.pt',
    data='libs/yolov5/data/coco128.yaml',
    device=device
)


for cls, label in enumerate(label_names):

    dataset = loadData(source=input_dir / label)
    for path, _, im0, _, _ in tqdm(dataset, desc=label):
        # im0 = im0s # HWC
        wh = im0.shape[:2][::-1]
        _, peoXyxyBoxes, confs, _ = peopleDec.detectSingleImg(im0, classes=[0], conf_thres=0.45)
        file = Path(path)
        if(len(peoXyxyBoxes) > 0) :
            arg = np.array(confs).argmax()
            box = np.array(xyxy2centerwh(peoXyxyBoxes[arg]))
            box[2:] /= wh
            box[:2] /= wh
            outputPath = label_dir[cls] / (Path(path).stem + ".txt")
            with outputPath.open('w') as f:
                f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

