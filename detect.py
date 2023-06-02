import argparse
import cv2
from pathlib import Path
import time

from libs.yolov5 import increment_path, print_args, LOGGER, check_requirements
from detectors import PhoneAndWalkDetector as PhoneWalkDetector
from IO import DataWriter, DataReader, YoloFileWriter as FileWriter


def saveCrop(saveDir, actionname, filename, crop):
    cropPath = increment_path(saveDir / 'crop'/ actionname / (filename+'.jpg'),sep='_')
    cropPath.parent.mkdir(parents=True, exist_ok=True)
    assert(cv2.imwrite(cropPath, crop))


def saveCrops(item, saveDir, labels):
    (crops, labelIds, filename) = item
    for i, crop in enumerate(crops):
        saveCrop(saveDir, labels[labelIds[i]], filename, crop)


def viewImg(item, delay):
    filename, img = item
    cv2.imshow(filename, img)
    cv2.waitKey(delay)


def wait(writer, info = None):
    while(writer.running()):
        time.sleep(1)
        if info is not None: LOGGER.info(info)

def run(
    source,
    device = 0,
    view_img = True,
    line_thickness = 2,
    nosave = False,
    save_crop = True,
    save_dir = 'runs/test',
    name = 'exp',
    exist_ok = False,
    vid_stride = 1
):
    
    # load data
    dataset = DataReader(source=source,batch_size=1, vid_stride=vid_stride).start()

    # load detector
    pwd = PhoneWalkDetector(device)

    # values about save 
    save = not nosave
    save_dir = increment_path(save_dir+'/'+name, exist_ok=exist_ok, mkdir=True) if save or save_crop else None
    preFilename = ''
    if save_crop:
        cropWriter = DataWriter(saveCrops).start(saveDir = save_dir, labels = pwd.class_names)
    if save:
        saveWriter = FileWriter().start()
    if view_img:
        viewWriter = DataWriter(viewImg).start(delay = -1)

    # time accumulator
    totalTime = 0.0
    capCount = 0

    # for per image or per frame(cap)
    while (datas := dataset.read()):
        path, im0, vid_info, infoStr, mode = datas[0] # HWC , BGR

        # get filename without suffix and suffix
        filename = Path(path).stem
        suffix = Path(path).suffix
        if preFilename != (filename+suffix):
            isNew = True
            preFilename = filename+suffix
        else:
            isNew = False

        # time recorder
        capCount += 1

        labelIds, _, _, crops, img, times = pwd.detectSingleImage(im0, conf_thres=0.5, mode = 'image', isNew=isNew, line_thickness = line_thickness)
        # labelIds, _, _, crops, img, times = pwd.detectSingleImage(im0, conf_thres=0.5, mode = mode, isNew=isNew, line_thickness = line_thickness)

        # print time
        LOGGER.info(f"{infoStr}\n      phonewalking detection :{'' if len(labelIds) else '(no detections), '}{sum(times) * 1E3:.1f}ms")
        LOGGER.info(f"      people detection :{times[0] * 1E3:.1f}ms")
        LOGGER.info(f"      pose estimation: {times[1] * 1E3:.1f}ms")
        LOGGER.info(f"      action estimation: {times[2] * 1E3:.1f}ms")
        LOGGER.info(f"      phone detection: {times[3] * 1E3:.1f}ms")
        totalTime += sum(times)

        if save_crop:
            cropWriter.save((crops, labelIds, filename))

        # save image/video
        if save:
            # videoWriter = saveImageOrVeido(save_dir / (filename + suffix), dataset.mode, img, videoWriter, vid_info, isNew)
            saveWriter.save(save_dir / (filename + suffix), mode, img, vid_info, isNew)

        # view image
        if view_img:  
            viewWriter.save((filename, img))

    # ending of for ---------------------------------------------------------------------------------
    LOGGER.info(f"{source}, total time: {totalTime:.1f}s, average process time: {totalTime / capCount * 1E3:.1f}ms")
    if save_crop:
        cropWriter.toEnd()
        wait(cropWriter, 'Rendering remaining ' + str(cropWriter.count()) + ' crops in the queue...\n')
    if save:
        saveWriter.toEnd()
        wait(saveWriter, 'saving...')
    if view_img:
        viewWriter.toEnd()
        wait(viewWriter)
    if save or save_crop: LOGGER.info(f"results save to {save_dir}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='datasets/testdata/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--nosave', action='store_true', help='save images/videos result')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-dir', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'), install=False)
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)