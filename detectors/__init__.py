import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from PhoneWalkDetector import PhoneWalkDetector
from YoloDetector import YoloDetector
from PhoneAndWalkDetector import PhoneWalkDetector as PhoneAndWalkDetector

class YoloPhoneWalkDetector(YoloDetector):
    def __init__(self, device):
        super().__init__(device, 'weights/yolov5/phoneWalk_yolo_ep500.pt')

class YoloPhoneActionEstimation(YoloDetector):
    def __init__(self, device):
        super().__init__(device, 'weights/yolov5/action_yolo_ep500.pt')

class YoloPhoneDetector(YoloDetector):
    def __init__(self, device):
        super().__init__(device, 'weights/yolov5/phoneEx5.pt')

__all__ = [
    PhoneWalkDetector, YoloPhoneWalkDetector, PhoneAndWalkDetector,
    YoloPhoneActionEstimation,
    YoloPhoneDetector
]