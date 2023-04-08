from libs.yolov5.yolov5DetectorApi import TargetsDetector
from libs.yolov5 import select_device

from Detector import Detector

class YoloPhoneWalkDetector(Detector):

    def __init__(self, device):
        device  = select_device(device)
        self.detector = TargetsDetector(
            weights='libs/yolov5/weights/phoneWalk.pt',
            data = 'libs/yolov5/data/phoneWalk.yaml',
            device=device
        )

    def detectSingleImage(self, im0, conf_thres):
        cls, targetBoxes, confs, time = self.detector.detectSingleImage(im0, conf_thres)
        return cls, targetBoxes, confs, [time, 0, 0, 0]
