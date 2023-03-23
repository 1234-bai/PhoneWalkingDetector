import argparse
import cv2
from pathlib import Path
import numpy as np

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator, select_device
from libs.yolov5.utils.plots import colors, save_one_box
from libs.yolov5.utils.general import check_requirements, increment_path, print_args
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from libs.st_gcn.StgcnApi import ActionEstimation as PhoneActionEstimation
from libs.st_gcn.TwoStreamStgcn import ActionEstimation as StandActionEstimation
from _utils.PointsUtils import pointsAnyInBox
from _utils.PoseTransformer import getBodyPartIndex, toBoneboxCoord, \
    coco2017Keypoints2CocoCut as co2cocut, coco2017Keypoints2openposeCoco as cO, nochange as hh


def loadModels(device):
    # people detector
    peoDt =  TargetsDetector(
        weights='D:/_NewCode/PythonPro/Phone_Walking_Detector/libs/yolov5/weights/yolov5s.pt',
        data='libs/yolov5/data/coco128.yaml',
        device=device
    )
    # phone detector
    phoneDt= TargetsDetector(
        weights='D:/_NewCode/PythonPro/Phone_Walking_Detector/libs/yolov5/weights/phone_ep20.pt',
        data='D:/_NewCode/PythonPro/Phone_Walking_Detector/libs/yolov5/data/phone.yaml',
        device=device
    )
    # people pose estimation
    poseEstimation = SingleImagePoseEstimation(device=device)
    # action estimation of holding phone with hand(s) 
    phoneAe = PhoneActionEstimation(device=device)
    # action estimation of sitting and standing
    walkAe = StandActionEstimation(device=device)

    return peoDt, phoneDt, poseEstimation, phoneAe, walkAe

def phoneWalkingActionAstimationOfSingleImage(phoneActionEstimation, walkingActionEstimation,  keypoints, score):
    '''
        keypoints: not normalizied skeleton keypoints
        score : confidence of keypoints
    '''
    kp = toBoneboxCoord(co2cocut(keypoints, [17, 2]), norm=True) # normalizied keypoints according to skeletion box
    sit = walkingActionEstimation.predictSingleCap(kp, co2cocut(score, [17,1]), None, normed=True)
    sit = walkingActionEstimation.getLabel(sit)
    print(sit)    
    if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
        return False
    kp = toBoneboxCoord(keypoints, norm=True)
    phone = phoneActionEstimation.predictSingleCap(kp, score, None, normed=True)
    phone = phoneActionEstimation.getLabel(phone)
    print(phone)    
    if phone in ['call', 'PlayWithOneHand', 'PlayWithTwoHands']:
        return True
    return False

def phoneInHandnotInEars(keypoints, poseFormat, phoneXyxy):
    wristpoints = keypoints[getBodyPartIndex(poseFormat, 'wrist')]
    earpoints = keypoints[getBodyPartIndex(poseFormat, 'ear')]
    if pointsAnyInBox(earpoints, phoneXyxy):
        return False
    if pointsAnyInBox(wristpoints, phoneXyxy, 5):
        return True
    return False

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
    source = 'data/images',
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
    # load device
    device = select_device(device)

    # load models
    peoDt, phoneDt, poseEstimation, phoneAe, walkAe = loadModels(device)
    
    # load data
    dataset = peoDt.loadData(source=source,vid_stride=vid_stride)

    save = not nosave
    saveDir = increment_path(save_dir+'/'+name, exist_ok=exist_ok, mkdir=True) if save or save_crop else None
    preFilename = ''
    # for per image or per frame(cap)
    for path, _, im0s, vid_cap, _ in dataset:

        # get filename without suffix and suffix
        if dataset.mode == 'stream':
            filename = path[0]
            suffix = '.mp4'
        else:
            filename = Path(path).stem
            suffix = Path(path).suffix

        # get original image
        im0 = im0s[0] if dataset.mode == 'stream' else im0s # HWC , BGR
        img = im0.copy()

        # annotator（drawer）
        annotator = TargetsAnnotator(img, line_thickness)

        # 检测人像
        _, peopleXyxyBoxes, crops, confs = peoDt.detectorSingleImg(img, classes=[0])
        if(len(peopleXyxyBoxes) > 0):   # 存在人像

            # if dataset.mode == 'image':
                # 根据人像检测骨骼结点
                poses = poseEstimation.process(img, peopleXyxyBoxes, confs) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number

                # 对每个骨骼结点进行手机检测和动作检测
                for i,pose in enumerate(poses): #   对于每个人像
                    peopleBox = peopleXyxyBoxes[i]  # 获得此人的人像盒子xyxy
                    keypoints = pose['keypoints'] # 获得此人的骨骼结点
                    score = pose['kp_score'] # 获得此人的骨骼结点置信度

                    # action estimation
                    if not phoneWalkingActionAstimationOfSingleImage(phoneAe, walkAe, keypoints, score):
                        continue
                    else:
                        actionName = 'phoneWalking'

                    # (手持)手机检测
                    _, phoneXyxyBoxes,_, _ = phoneDt.detectorSingleImg(crops[i], classes=[0], conf_thres=0.4)
                    if(len(phoneXyxyBoxes)):   # 人像图中存在手机
                        for phoneBox in phoneXyxyBoxes:  # 对于每个手机，是否与人手重合
                            phoneBox += np.array(peopleBox)[[0, 1, 0, 1]]
                            if phoneInHandnotInEars(keypoints, poseEstimation.poseType, phoneBox):
                                if save_crop:
                                    cropPath = increment_path(saveDir / 'crop'/ (filename+'.jpg'),sep='_')
                                    save_one_box(peopleBox, im0, file=cropPath, BGR=True)
                                annotator.box_label(peopleBox, label=actionName, color=colors(0))   # 深拷贝，会直接在原始图片上进行修改
                                annotator.box_label(phoneBox, label='phone', color=colors(5))
                                break

        img = annotator.result()
        filename = filename + suffix
        # save image/video
        if save:
            videoWriter = saveImageOrVeido(saveDir / filename, dataset.mode, img, videoWriter, vid_cap, filename != preFilename)
            preFilename = filename

        # view image
        if view_img:  
            cv2.imshow(filename, img)  
            if cv2.waitKey(-1) & 0xFF == ord('q'):
                break





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--nosave', action='store_true', help='save images/videos result')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-dir', default='runs/test', help='save results to project/name')
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