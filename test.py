from tqdm import tqdm
import argparse
import torch

from libs.yolov5 import (
        ConfusionMatrix, LoadImagesAndLabels, 
        increment_path, print_args, check_requirements, LOGGER
    )
from detectors import PhoneWalkDetector, YoloPhoneWalkDetector


class Model:
    def __init__(self, model_type, device) -> None:
        self.model_type = model_type
        self.model =  YoloPhoneWalkDetector(device) if model_type == 'yolo' else PhoneWalkDetector(device)
        pass

    def detectSingleImage(self, im0, conf_thres):
        if self.model_type == 'yolo':
            labelIds, targetBoxes, confs, time = self.model.detectSingleImage(im0, conf_thres)
        else:
            labelIds, targetBoxes, confs, _, _, time = self.model.detectSingleImage(im0, 'image', False, conf_thres)
        labelIds = torch.tensor(labelIds)
        confs = torch.tensor(confs)
        targetBoxes = torch.tensor(targetBoxes)
        detections = torch.cat((targetBoxes, confs.unsqueeze(1), labelIds.unsqueeze(1)), dim=1)
        return detections, sum(time)

def run(
    device, 
    model_type,
    images_path,
    save_dir,
    name
):
    dataset = LoadImagesAndLabels(images_path)
    pwd = Model(model_type, device=device)
    class_count = 5 # (call, one, two, stand, other)
    cMatrix = ConfusionMatrix(class_count)
    totalTime = 0.0
    for im, targetLabels, path in tqdm(dataset, desc=images_path):
        allDetections, time = pwd.detectSingleImage(im.numpy(), conf_thres=0.5)
        totalTime += time
        detectionLabels = allDetections[:, 5] if len(allDetections) else torch.tensor([])
        trueLabels = targetLabels[:, 0]
        for i in range(class_count):
            detections = allDetections[detectionLabels == i]
            labels = targetLabels[trueLabels == i]
            cMatrix.process_batch(detections, labels)
    save_dir=increment_path(save_dir+'/'+name, mkdir=True)
    cMatrix.plot(save_dir=save_dir, normalize=False, names=['call', 'one', 'two', 'stand', 'other'])
    LOGGER.info(f'sum:{cMatrix.matrix.sum()}')
    LOGGER.info(f'process time for per image: {(totalTime / dataset.n) * 1E3:.2f}ms')
    LOGGER.info(f'results have be saved to {save_dir}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, default='datasets/yolodata/val.txt', help='path to images files/directory')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model', default='pwd', dest = 'model_type',help='test modeltype, i.e. yolo or pwd')
    parser.add_argument('--save-dir', default='runs/test', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)