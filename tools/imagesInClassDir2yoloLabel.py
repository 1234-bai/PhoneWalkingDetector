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
from utils.PointsUtils import xyxy2centerwh


input_dir = Path("datasets/yolodata/train_val/images")
output_dir = Path('datasets/yolodata/train_val/labels')

label_names = ['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'Stand', 'Sit'] # 根据动作分类，而不是手机出现的位置
label_map = [1, 1, 2, 0, 3]
label_dir = [(output_dir / x) for x in label_names]
for x in label_dir:
    x.mkdir(parents=True, exist_ok=True)

device = select_device(0)
peopleDec =  TargetsDetector(
    weights='weights/yolov5/yolov5s.pt',
    device=device
)

for i, label in enumerate(label_names):
    cls = label_map[i]
    dataset = loadData(source=input_dir / label)
    for path, im0, _, _ in tqdm(dataset, desc=label):
        # im0 = im0s # HWC
        wh = im0.shape[:2][::-1]
        _, peoXyxyBoxes, confs, _ = peopleDec.detectSingleImage(im0, classes=[0], conf_thres=0.45)
        file = Path(path)
        if(len(peoXyxyBoxes) > 0) :
            for peoBox in peoXyxyBoxes:
                box = np.array(xyxy2centerwh(peoBox))
                box[2:] /= wh
                box[:2] /= wh
                if box[2] > 0.6 and box[3] > 0.6:
                    outputPath = label_dir[i] / (Path(path).stem + ".txt")
                    with outputPath.open('w') as f:
                        f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

