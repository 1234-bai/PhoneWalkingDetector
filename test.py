from tqdm import tqdm
import argparse
import torch

from libs.yolov5 import (
        ConfusionMatrix, LoadImagesAndLabels, 
        increment_path, print_args, check_requirements, LOGGER
    )
from PhoneWalkDetector import PhoneWalkDetector

def run(
    device, 
    label_path,
    save_dir,
    name
):
    dataset = LoadImagesAndLabels(label_path)
    pwd = PhoneWalkDetector(device=device)
    class_count = 5 # (call, one, two, stand, other)
    cMatrix = ConfusionMatrix(class_count)
    # j = 0
    for im, targetLabels, path in tqdm(dataset, desc=label_path):
        # j += 1
        # if j > 5: break
        labelIds, targetBoxes, confs, _, _, _ = pwd.detectSingleImage(im.numpy(), 'image', False, conf_thres=0.0)
        labelIds = torch.tensor(labelIds)
        confs = torch.tensor(confs)
        targetBoxes = torch.tensor(targetBoxes)
        for i in range(class_count):
            deIndex = labelIds == i
            detections = torch.cat((targetBoxes[deIndex], confs[deIndex].unsqueeze(1), labelIds[deIndex].unsqueeze(1)), dim=1) if sum(deIndex) else None
            labels = targetLabels[targetLabels[:, 0] == i]
            if len(labels) > 0: 
                cMatrix.process_batch(detections, labels)
    save_dir=increment_path(save_dir+'/'+name, mkdir=True)
    cMatrix.plot(save_dir=save_dir, normalize=False, names=['call', 'one', 'two', 'stand', 'other'])
    LOGGER.info(f'results have be saved to {save_dir}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-path', type=str, default='datasets/yolodata/val.txt', help='path to label file/directory')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', default='runs/test', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)