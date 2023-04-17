from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, save_one_box, select_device

from Detector import Detector

class YoloPhoneWalkDetector(Detector):

    def __init__(self, device):
        super().__init__(['call', 'playWithOneHand', 'playWithTwoHands', 'walking', 'other'])
        device  = select_device(device)
        self.detector = TargetsDetector(
            weights='weights/yolov5/phoneWalk.pt',
            data = 'libs/yolov5/data/phoneWalk.yaml',
            device=device
        )

    def detectSingleImage(self, im0, conf_thres, mode = 'image', isNew = True, line_thickness = 2):
        cls, targetBoxes, confs, time = self.detector.detectSingleImage(im0, conf_thres)
        crops = []
        img = im0
        if len(targetBoxes):
            annotator = TargetsAnnotator(im0, 2)
            for i, box in enumerate(targetBoxes):
                annotator.box_label(box, self.detector.getLabelName(cls[i])+str(round(float(confs[i]), 2)), colors(i))
                crops.append(save_one_box(box, im0, save=False, BGR=True))
            img = annotator.result()
        return cls, targetBoxes, confs, crops, img, [time, 0, 0, 0]
