import sys
from pathlib import Path
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.torch_utils import select_device
from utils.plots import colors, save_one_box
from utils.general import check_file, check_requirements, increment_path, print_args, LOGGER, Profile
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from LoadImageAndLabels import LoadImagesAndLabels
from utils.metrics import ConfusionMatrix, box_iou, ap_per_class


def loadData(
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
    try:
        if webcam:
            dataset = LoadStreams(source, vid_stride=vid_stride)
        elif screenshot:
            dataset = LoadScreenshots(source)
        else:
            dataset = LoadImages(source, vid_stride=vid_stride)
    except FileNotFoundError as e:
        LOGGER.info('\n'+str(e))
        exit()

    return dataset


def getCorrectPredictionMatrix(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros((detections.shape[0], iouv.shape[0]), dtype=torch.bool)
    if len(detections):
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
    return correct.to(iouv.device)

__all__ = [
    select_device, loadData, 
    colors, save_one_box, increment_path, 
    LOGGER, Profile, print_args, check_requirements,
    ConfusionMatrix, LoadImagesAndLabels,
    getCorrectPredictionMatrix, ap_per_class
]