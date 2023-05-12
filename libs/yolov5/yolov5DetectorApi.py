# YOLOv5 APIs 🚀 by QianXmY, GPL-3.0 license

import numpy as np
import torch

from models.experimental import attempt_load
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator
from utils.augmentations import letterbox


class TargetsDetector:

    def __init__(
        self,
        weights,  # model path or triton URL
        device : torch.device,
        imgsz=(640, 640),  # inference size (height, width)
    ) -> None:
        # Load model
        # model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        model = attempt_load(weights, device=device)
        model.float()
        self.stride=  max(int(model.stride.max()), 32)
        names = model.module.names if hasattr(model, 'module') else model.names # stride表示的即是模型下采样次数的2的次方，这个涉及感受野的问题，在YOLOV5中下采样次数为5;names目标检测出的类别名字数组
        self.names = [names[i] for i in names] if isinstance(names, dict) else names
        self.model = model
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.device = device
        if device.type != 'cpu': # warmup
            im = torch.empty((1, 3, *imgsz), dtype= torch.float, device=device)  # input
            self.model(im)
        self.dt = (Profile(), Profile(), Profile())

    def detectSingleImage(
        self, 
        im0,
        conf_thres=0.5,  # confidence threshold 置信度阈值
        # iou_thres=0.45,  # NMS IOU threshold 非极大值抑制的交并比阈值
        classes=None,  # filter by class: --class 0, or --class 0 2 3 类别过滤器，只识别给出的类别编号，和配置文件中的类别编号相对应
        max_det=1000,  # maximum detections per image 每张图片目标检测的最大数量
    ):

        im = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # 图片像素点归一化
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32 
            im /= 255  # 0 - 255 to 0.0 - 1.0
            im = im[None]  # expand for batch dim

        # 预测
        with self.dt[1]:
            pred = self.model(im)[0]

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres=0.45, classes = classes, agnostic = False, max_det=max_det)

        results = [[], [], []]
        # imc = im0.copy()  # for save_crop
        det = pred[0]  # per image，对于每张图片
        # det : N * (x1, y1, x2, y2, conf, class)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det): # 对于每张图片的多个目标
                # crop = save_one_box(xyxy, imc, save=False, BGR=True)
                results[0].append(int(cls.cpu()))
                results[1].append(torch.tensor(xyxy).tolist())
                # results[2].append(crop)
                results[2].append(float(conf.cpu()))
                # results.append((c, torch.tensor(xyxy).tolist(), crop, conf))

        return results[0], results[1], results[2], self.dt[1].dt + self.dt[0].dt + self.dt[2].dt
    

    def detectImages(
        self, 
        im0s,
        conf_thres=0.5,  # confidence threshold 置信度阈值
        # iou_thres=0.45,  # NMS IOU threshold 非极大值抑制的交并比阈值
        classes=None,  # filter by class: --class 0, or --class 0 2 3 类别过滤器，只识别给出的类别编号，和配置文件中的类别编号相对应
        max_det=1000,  # maximum detections per image 每张图片目标检测的最大数量
    ):
        ims = []
        for im in im0s:
            im = letterbox(im, self.imgsz, stride=self.stride, auto=False, scaleFill=True)[0]
            im = im.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
            ims.append(im)
        ims = np.ascontiguousarray(np.array(ims))

        # 图片像素点归一化
        with self.dt[0]:
            ims = torch.from_numpy(ims).to(self.device)
            ims = ims.float()  # uint8 to fp16/32 
            ims /= 255  # 0 - 255 to 0.0 - 1.0

        # 预测
        with self.dt[1]:
            pred = self.model(ims)[0]

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres=0.45, classes = classes, agnostic = False, max_det=max_det)

        results = []
        # imc = im0.copy()  # for save_crop
        for i, det  in enumerate(pred):   # per image，对于每张图片
            # det : N * (x1, y1, x2, y2, conf, class)
            result = ([], [], [])
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(ims[i].shape[1:], det[:, :4], im0s[i].shape[1:]).round()

                for *xyxy, conf, cls in reversed(det): # 对于每张图片的多个目标
                    # crop = save_one_box(xyxy, imc, save=False, BGR=True)
                    result[0].append(int(cls.cpu()))
                    result[1].append(torch.tensor(xyxy).tolist())
                    # results[2].append(crop)
                    result[2].append(float(conf.cpu()))
                    # results.append((c, torch.tensor(xyxy).tolist(), crop, conf))
            results.append(result)

        return results, self.dt[1].dt + self.dt[0].dt + self.dt[2].dt

    def getLabelName(self, c):
        return self.names[c]


class TargetsAnnotator(Annotator) :
    
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        super().__init__(im, line_width, font_size, font, pil, example)


    def double_box_label(self, box1 : list, box2 : list, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        box = np.array(box1)[[0, 1, 0, 1]]
        box = (box + np.array(box2)).tolist()
        return super().box_label(box, label, color, txt_color)
    

