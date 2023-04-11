import argparse
import cv2
from pathlib import Path

from libs.yolov5 import increment_path, print_args, LOGGER, loadData, check_requirements
from detectors.PhoneWalkDetector import PhoneWalkDetector


def saveCrop(saveDir, actionname, filename, crop):
    cropPath = increment_path(saveDir / 'crop'/ actionname / (filename+'.jpg'),sep='_')
    cropPath.parent.mkdir(parents=True, exist_ok=True)
    assert(cv2.imwrite(cropPath, crop))


def saveImageOrVeido(savePath, mode, img, videoWriter, videoCap, isNew):
    '''
        img : HWC, BGR
    '''
    if mode == 'image':
        cv2.imwrite(savePath, img)
    else:   # stream or vedio
        if isNew: # 是一个新的视频或者第一个视频
            if videoWriter is not None:
                videoWriter.release() # release previous video writer
                videoWriter = None
            if videoCap: #video
                fps = videoCap.get(cv2.CAP_PROP_FPS)
                w = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:   # stream
                fps, w, h = 30, img.shape[1], img.shape[0]
            videoWriter = cv2.VideoWriter(str(savePath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        assert(videoWriter != None)
        videoWriter.write(img) # 是前一个视频的下一帧
        return videoWriter


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
    dataset = loadData(source=source,vid_stride=vid_stride)

    # load detector
    pwd = PhoneWalkDetector(device)

    # values about save 
    save = not nosave
    save_dir = increment_path(save_dir+'/'+name, exist_ok=exist_ok, mkdir=True) if save or save_crop else None
    preFilename = ''
    videoWriter = None

    # time accumulator
    totalTime = 0.0
    capCount = 0

    # for per image or per frame(cap)
    for path, im0s, vid_cap, infoStr in dataset:

        # get filename without suffix and suffix
        if dataset.mode == 'stream':
            filename = path[0]
            suffix = '.mp4'
        else:
            filename = Path(path).stem
            suffix = Path(path).suffix
        if preFilename != (filename+suffix):
            isNew = True
            preFilename = filename+suffix
        else:
            isNew = False

        # get original image
        im0 = im0s[0] if dataset.mode == 'stream' else im0s # HWC , BGR

        # time recorder
        capCount += 1

        labelIds, _, _, crops, img, times = pwd.detectSingleImage(im0, dataset.mode, isNew, line_thickness = line_thickness)

        # print time
        LOGGER.info(f"{infoStr}\n      people detection :{'' if len(labelIds) else '(no detections), '}{times[0] * 1E3:.1f}ms")
        LOGGER.info(f"      pose estimation: {times[1] * 1E3:.1f}ms")
        LOGGER.info(f"      action estimation: {times[2] * 1E3:.1f}ms")
        LOGGER.info(f"      phone detection: {times[3] * 1E3:.1f}ms")
        totalTime += sum(times)

        if save_crop:
            for i, crop in enumerate(crops):
                saveCrop(save_dir, pwd.getLabel(labelIds[i]), filename, crop)

        # save image/video
        if save:
            videoWriter = saveImageOrVeido(save_dir / (filename + suffix), dataset.mode, img, videoWriter, vid_cap, isNew)

        # view image
        if view_img:  
            cv2.imshow(filename, img)
            if cv2.waitKey(-1) & 0xFF == ord('q'):
                break

    # ending of for ---------------------------------------------------------------------------------

    LOGGER.info(f"{source}, total time: {totalTime * 1E3:.1f}, average process time: {totalTime / capCount * 1E3:.1f}ms")
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
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)