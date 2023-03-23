# YOLOv5 APIs 🚀 by QianXmY, GPL-3.0 license

import sys
from pathlib import Path
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, save_one_box
from utils.torch_utils import select_device
from utils.augmentations import letterbox


class TargetsDetector:

    def __init__(
            self,
            weights,  # model path or triton URL
            data,  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            dnn=False,  # use OpenCV DNN for ONNX inference
            half=False,  # use FP16 half-precision inference
            device : torch.device = 0  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    ) -> None:
        # Load model
        # device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

        self.stride, self.names, self.pt = model.stride, model.names, model.pt # stride表示的即是模型下采样次数的2的次方，这个涉及感受野的问题，在YOLOV5中下采样次数为5;names目标检测出的类别名字数组
        self.model = model
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.warmedupFlag = False
        self.dt = (Profile(), Profile(), Profile())

    def loadData(
        self,
        source,  # file/dir/URL/glob/screen/0(webcam)
        vid_stride=1  # video frame-rate stride 帧率：视频每多少帧截取一次):
    ):
        # 区分数据来源 
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
        
        # 加载数据
        if webcam:
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)

        return dataset


    def detectorSingleImg(
        self, 
        im0,
        augment=False,  # augmented inference
        conf_thres=0.5,  # confidence threshold 置信度阈值
        iou_thres=0.45,  # NMS IOU threshold
        classes=None,  # filter by class: --class 0, or --class 0 2 3 类别过滤器，只识别给出的类别编号，和配置文件中的类别编号相对应
        agnostic_nms=False,  # class-agnostic NMS
        max_det=1000,  # maximum detections per image 每张图片目标检测的最大数量
    ):

        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        results = [[], [], [], []]
        # 图片像素点归一化
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        if self.warmedupFlag == False:
            self.model.warmup(imgsz=(1, 3, *(self.imgsz)))  # 模型使用前预处理
            self.warmedupFlag = True

        # 预测
        with self.dt[1]:
            pred = self.model(im, augment=augment, visualize=False)  

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 一张图片目标检测后会有多类目标被识别出来
        imc = im0.copy()  # for save_crop
        for i, det in enumerate(pred):  # per image，对于每类识别出来的目标
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 每类识别目标有多个
                for *xyxy, conf, cls in reversed(det): # 对于特定一类检测目标的多个
                    crop = save_one_box(xyxy, imc, save=False, BGR=True)
                    results[0].append(int(cls))
                    results[1].append(torch.tensor(xyxy).tolist())
                    results[2].append(crop)
                    results[3].append(conf)
                    # results.append((c, torch.tensor(xyxy).tolist(), crop, conf))

        # Print time (inference-only)
        LOGGER.info(f"{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

        return results[0], results[1], results[2], results[3]
    
    def getLabelName(self, c):
        return self.names[c]


class TargetsAnnotator(Annotator) :
    
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        super().__init__(im, line_width, font_size, font, pil, example)


    def double_box_label(self, box1 : list, box2 : list, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        box = np.array(box1)[[0, 1, 0, 1]]
        box = (box + np.array(box2)).tolist()
        return super().box_label(box, label, color, txt_color)
    

