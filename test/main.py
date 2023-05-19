import argparse
import cv2
from pathlib import Path
import numpy as np
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:/_NewCode/PythonPro/Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import (
        colors, save_one_box, check_requirements, increment_path, print_args, select_device, 
        LOGGER, Profile,
        loadData
    )
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from libs.st_gcn.StgcnApi import ActionEstimation as HandActionEstimation
from libs.st_gcn.TwoStreamStgcn import ActionEstimation as StandActionEstimation
from utils.PointsUtils import pointsAnyInBox, xywh2xyxy
from utils.PoseTransformer import getBodyPartIndex, toBoneboxCoord, coco2017Keypoints2CocoCut as co2cocut


def loadModels(device):
    # people detector
    peoDt =  TargetsDetector(
        weights='weights/yolov5/yolov5s.pt',
        device=device
    )
    # phone detector
    phoneDt= TargetsDetector(
        weights='weights/yolov5/phoneEp80.pt',
        device=device
    )
    # people pose estimation
    poseEstimation = SingleImagePoseEstimation(device=device)
    # action estimation of holding phone with hand(s) 
    phoneAe = HandActionEstimation(
        weight_file='weights/stgcn/stgcn_class3_150_94_ex9.pt',
        class_names=['nohand', 'oneHand', 'twoHands'],
        device=device
    )
    # action estimation of sitting and standing
    walkAe = StandActionEstimation(device=device)

    return peoDt, phoneDt, poseEstimation, phoneAe, walkAe

def phoneWalkingAeOfSingleImage(phoneActionEstimation, walkingActionEstimation,  keypoints, score, dt : Profile):
    '''
        phoneWalking Action Astimation Of Single Image
        params:
            keypoints: not normalizied skeleton keypoints
            score : confidence of keypoints
    '''
    kp = toBoneboxCoord(co2cocut(keypoints, [17, 2]), norm=True) # normalizied keypoints according to skeletion box
    sit, sitTime = walkingActionEstimation.predictSingleCap(kp, co2cocut(score, [17,1]), None, normed=True)
    dt.t += sitTime
    sit = walkingActionEstimation.getLabel(sit)  
    if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
        return False
    kp = toBoneboxCoord(keypoints, norm=True) - 0.5    # normalization and centralization
    phone, phoneTime = phoneActionEstimation.predictSingleCap(kp, score, None, normed=True)
    dt.t += phoneTime
    phone = phoneActionEstimation.getLabel(phone)
    if phone == 'nohand':
        return False
    return True

def phoneWalkingAeOfMultiCaps(phoneActionEstimation, walkingActionEstimation,  tvc, dt : Profile):
    '''
        phoneWalking Action Astimation Of Single Image
        params:
            keypoints: not normalizied skeleton keypoints
            score : confidence of keypoints
            tvc:  points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).
                
    '''
    tvc = np.array(tvc)
    cococutTvc = []
    for vc in tvc:
        vc = co2cocut(vc, [17, 3])
        vc[:,:2] = toBoneboxCoord(vc[:,:2], norm=True)
        cococutTvc.append(vc)
    sit, sitTime = walkingActionEstimation.predict(np.array(cococutTvc), None, normed=True)
    dt.t += sitTime
    sit = walkingActionEstimation.getLabel(sit) 
    if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
        return False
    for i,vc in enumerate(tvc):
        tvc[i][:,:2] = toBoneboxCoord(vc[:,:2], norm=True) - 0.5    # normalization and centralization
    phone, phoneTime = phoneActionEstimation.predict(tvc, None, normed=True)
    phone = phoneActionEstimation.getLabel(phone)
    dt.t += phoneTime
    if phone == 'nohand':
        return False
    return True, sitTime + phoneTime

def phoneInHand(keypoints, poseFormat, phoneXyxy):
    wristpoints = keypoints[getBodyPartIndex(poseFormat, 'wrist')]
    if pointsAnyInBox(wristpoints, phoneXyxy, 0.75):
        return True
    return False

def phoneInEars(keypoints, poseFormat, phoneXyxy):
    earpoints = keypoints[getBodyPartIndex(poseFormat, 'ear')]
    if pointsAnyInBox(earpoints, phoneXyxy, 0):
        return True
    return False


def playPhoneDetection(phoneDetecter, peoCrop, cropBox, peoKeypoints, poseType, dt):
    '''
        param:
            cropBox: people crop coord based img box
        return:
            phone box based img box
    '''
    # (手持)手机检测
    _, phoneXyxyBoxes,_, time = phoneDetecter.detectSingleImage(peoCrop, classes=[0], conf_thres=0.4)
    dt.t += time
    pB = None
    action = ''
    if(len(phoneXyxyBoxes)):   # 人像图中存在手机
        for phoneBox in phoneXyxyBoxes:  # 对于每个手机，是否与人手重合
            phoneBox += np.array(cropBox)[[0, 1, 0, 1]]
            if phoneInHand(peoKeypoints, poseType, phoneBox):
                if phoneInEars(peoKeypoints, poseType, phoneBox):
                    pB = phoneBox
                    action = 'call'
                else:
                    return phoneBox, 'playphone'
    return pB, action


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

def drawPlayphoneAndCall(annotator, peopleBox, phoneBox, actionName):
    red_bgr = (0, 0, 256)
    black_bgr = (0, 0, 0)
    color = red_bgr if actionName == 'playphone' else colors(0)
    annotator.box_label(peopleBox, label=actionName, color=color)   # 深拷贝，会直接在原始图片上进行修改
    annotator.box_label(phoneBox, label='phone', color=colors(5), txt_color=black_bgr)

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
    dataset = loadData(source=source,vid_stride=vid_stride)

    # values about save 
    save = not nosave
    saveDir = increment_path(save_dir+'/'+name, exist_ok=exist_ok, mkdir=True) if save or save_crop else None
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
        img = im0.copy()

        # time recorder
        dt = [Profile(), Profile()]
        peTime = 0.0
        capCount += 1

        # 检测人像
        _, peopleXyxyBoxes, confs, time = peoDt.detectSingleImage(img, classes=[0])
        
        if(len(peopleXyxyBoxes) > 0):   # 存在人像

            # annotator（drawer）
            annotator = TargetsAnnotator(img, line_thickness)

            if dataset.mode == 'image':

                # 根据人像检测骨骼结点
                poses, peTime = poseEstimation.process(img, peopleXyxyBoxes, confs) # 获得骨骼结点 

                # 对每个骨骼结点进行手机检测和动作检测
                for pose in poses: #   对于每个人像

                    peopleBox = xywh2xyxy(pose['bbox'])  # 获得此人的人像盒子xyxy
                    keypoints = pose['keypoints'] # 获得此人的骨骼结点
                    score = pose['kp_score'] # 获得此人的骨骼结点置信度

                    # action estimation
                    if not phoneWalkingAeOfSingleImage(phoneAe, walkAe, keypoints, score, dt[0]):
                        continue

                    # (手持)手机检测
                    crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                    phoneBox, actionName = playPhoneDetection(phoneDt, crop, peopleBox, keypoints, poseEstimation.poseType, dt[1])
                    if phoneBox is not None:
                        drawPlayphoneAndCall(annotator, peopleBox, phoneBox, actionName)
                        if save_crop:
                            saveCrop(saveDir, actionName, filename, crop)
            else:

                if isNew: # new video or first video
                    poseEstimation.initTracker()
                    poseStore = []  # peos in last frame
                
                poses, peTime = poseEstimation.process(im0, peopleXyxyBoxes, confs, tracking=True)

                existedPeo = ([], {}) # peos in now frame
                for ps in poses:
                    id = ps['idx']
                    existedPeo[0].append(id)
                    existedPeo[1][id] = xywh2xyxy(ps['bbox'])
                    kp = ps['keypoints']
                    vc = np.concatenate((kp,ps['kp_score']), axis=1)
                    try:
                        poseStore[id].append(vc)
                    except IndexError:
                        poseStore.append([vc])

                for id,tvc in enumerate(poseStore):
                    if id in existedPeo[0]:
                        actionName = 'pending'
                        if len(tvc) > 1:
                            if not phoneWalkingAeOfMultiCaps(phoneAe, walkAe, tvc, dt[0]):
                                continue
                            peopleBox = existedPeo[1][id]
                            crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                            phoneBox, actionName = playPhoneDetection(phoneDt, crop, peopleBox, tvc[-1][:,:2], poseEstimation.poseType, dt[1])
                            if phoneBox is not None:
                                drawPlayphoneAndCall(annotator, peopleBox, phoneBox, actionName)
                                if save_crop:
                                    saveCrop(saveDir, actionName, filename, crop)
                    else:
                        tvc.clear()

            # end of img/video-cap process --------------------------------------------------------------------------------

            img = annotator.result()
        
        # end of people detector -----------------------------------------------------------------------

        # print time
        LOGGER.info(f"{infoStr}\n      people detection :{'' if len(peopleXyxyBoxes) else '(no detections), '}{time * 1E3:.1f}ms")
        LOGGER.info(f"      pose estimation: {peTime * 1E3:.1f}ms")
        LOGGER.info(f"      action estimation: {dt[0].t * 1E3:.1f}ms")
        LOGGER.info(f"      phone detection: {dt[1].t * 1E3:.1f}ms")
        totalTime += (time + peTime + dt[0].t + dt[1].t)


        # save image/video
        if save:
            videoWriter = saveImageOrVeido(saveDir / (filename + suffix), dataset.mode, img, videoWriter, vid_cap, isNew)

        # view image
        if view_img:  
            cv2.imshow(filename, img)  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ending of for ---------------------------------------------------------------------------------

    LOGGER.info(f"{source}, average process time: {totalTime / capCount * 1E3:.1f}ms")



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
    check_requirements(install=False, exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)