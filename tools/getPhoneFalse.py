import numpy as np
import cv2
import sys
from pathlib import Path
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, select_device, LOGGER, loadData
from _utils.PointsUtils import xyxy2centerwh

phoneTest= TargetsDetector(
    weights='weights/yolov5/phoneEp80.pt',
    device=select_device(0)
)


def run(
    input_dir,
    out_put_dir
):
    images_dir = (Path(out_put_dir) / 'images')
    images_dir.mkdir(exist_ok=True)
    labels_dir = (Path(out_put_dir) / 'labels')
    labels_dir.mkdir(exist_ok=True)
    dataset = loadData(source=input_dir)
    for path, im0s, _, _ in tqdm(dataset, desc=input_dir):
        im0 = im0s[0] if dataset.mode == "stream" else im0s
        path = Path(path)
        filename = path.stem
        suffix = path.suffix
        wh = im0.shape[:2][::-1]
        cls, phoneXyxyBoxes, _, _ = phoneTest.detectSingleImage(im0, conf_thres=0.2)
        if len(phoneXyxyBoxes):
            assert(cv2.imwrite(str(images_dir / (filename+suffix)), im0))
            for i, box in enumerate(phoneXyxyBoxes):
                box = np.array(xyxy2centerwh(box))
                box[2:] /= wh
                box[:2] /= wh
                cls_str = '1 ' if cls[i] < 0.6 else '0 '
                with (labels_dir / (filename+".txt")).open('w') as f:
                    f.write(cls_str + ' '.join(map(str, box)) + "\n")

if __name__ == "__main__":
    run(
        input_dir = "D:/QianXiaoYi/Pictures/Data/phone/train_with_anoations/1_no_phone",
        out_put_dir = "datasets/phoneData/phoneExtension"
    )