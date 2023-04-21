from tqdm import tqdm
import argparse
import cv2
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5 import (
        LoadImagesAndLabels, print_args
    )
from libs.yolov5.yolov5DetectorApi import TargetsAnnotator

def run(
    images_path,
    classes = ['call', 'one', 'two', 'walk', 'other']
):
    dataset = LoadImagesAndLabels(images_path)
    for im, targetLabels, path in tqdm(dataset, desc=images_path):
        anonatator = TargetsAnnotator(im.numpy())
        for l in targetLabels:
            anonatator.box_label(l[1:], classes[int(l[0])])
        img = anonatator.result()
        cv2.imshow(Path(path).stem, img)
        if cv2.waitKey(-1) & 0xFF == ord('q'):
            break

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, default='datasets/yolodata/test/test.txt', help='path to images files/directory')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)