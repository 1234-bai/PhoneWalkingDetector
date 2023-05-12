from tqdm import tqdm
import argparse
import torch
import numpy as np

from libs.yolov5 import (
        ConfusionMatrix, LoadImagesAndLabels, getCorrectPredictionMatrix, ap_per_class,
        increment_path, print_args, check_requirements, LOGGER
    )
from detectors import (
    PhoneAndWalkDetector, PhoneWalkDetector, YoloPhoneWalkDetector,
    YoloPhoneActionEstimation
)
from detectors.StgcnActionEstimation import PhoneActionEstimation


class Model:
    def __init__(self, model_type, device) -> None:
        if model_type == 'yolo':
            self.model = YoloPhoneWalkDetector(device)
        elif model_type == 'pwd':
            self.model = PhoneWalkDetector(device)
        elif model_type == 'pawd':
            self.model = PhoneAndWalkDetector(device, 2)
        elif model_type == 'pawd_all':
            self.model = PhoneAndWalkDetector(device, 1)
        elif model_type == 'action_yolo':
            self.model = YoloPhoneActionEstimation(device)
        elif model_type == 'action_stgcn_single':
            self.model = PhoneActionEstimation(device, 1)
        elif model_type == 'action_stgcn_double':
            self.model = PhoneActionEstimation(device, 2)
        else:
            raise


    def detectSingleImage(self, im0, conf_thres):
        labelIds, targetBoxes, confs, _, _, time = self.model.detectSingleImage(im0, conf_thres, 'image', False)
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
    name,
    classes = ['call', 'one', 'two', 'walk', 'other'],
    norm = False,
):
    dataset = LoadImagesAndLabels(images_path)
    pwd = Model(model_type, device=device)
    iouv = torch.linspace(0.5, 0.95, 10)
    class_count = len(classes) # (call, one, two, stand, other)
    cMatrix = ConfusionMatrix(class_count)
    totalTime = 0.0
    stats = []
    for im, targetLabels, path in tqdm(dataset, desc=images_path):
        allDetections, time = pwd.detectSingleImage(im.numpy(), conf_thres=0.5)
        totalTime += time
        detections = allDetections if len(allDetections) else torch.tensor([])
        cMatrix.process_batch(detections, targetLabels)
        correct = getCorrectPredictionMatrix(detections, targetLabels, iouv)
        if len(detections):
            stats.append((correct, detections[:, 4], detections[:, 5], targetLabels[:, 0]))  # (correct, conf, pcls, tcls)
        else:
            stats.append((correct, *torch.zeros((2, 0)), targetLabels[:, 0]))

    save_dir=increment_path(save_dir+'/'+name, mkdir=True)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=classes)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=class_count)  # number of targets per class

    # Print/Save results
    with open(save_dir / 'result.txt', "w") as f:

        s = ('%11s' * 6) % ('Class', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        f.write(s+'\n')
        pf = '%11s' + '%11i' + '%11.3g' * 4  # print format
        s = pf % ('all', nt.sum(), mp, mr, map50, map)
        f.write(s+'\n')
        # Print results per class
        if class_count < 50 and class_count > 1 and len(stats):
            for i, c in enumerate(ap_class):
                s = pf % (classes[c], nt[c], p[i], r[i], ap50[i], ap[i])
                f.write(s+'\n')

        # Print speeds
        s = f'total time: {totalTime:.2f}s,process time for per image: {(totalTime / dataset.n) * 1E3:.2f}ms'
        f.write(s+'\n')

    # plot ConfusionMatrix
    cMatrix.plot(save_dir=save_dir, normalize=norm, names=classes)

    LOGGER.info(f'results have be saved to {save_dir}')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, default='datasets/yolodata/test/test.txt', help='path to images files/directory')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model', default='pwd', dest = 'model_type',help='test modeltype, i.e. yolo or pwd')
    parser.add_argument('--classes', nargs='+', type=str, default=['call', 'one', 'two', 'walk', 'other'], help='label names: --classes call walk')
    parser.add_argument('--save-dir', default='runs/test', help='save results to project/name')
    parser.add_argument('--name', default='exp_new_label_phoneWalk_nl', help='save results to project/name')
    parser.add_argument('--norm', default=False, action='store_true', help='normalize confusion matrix')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)