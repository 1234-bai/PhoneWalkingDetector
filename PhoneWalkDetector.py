import argparse
import cv2
from pathlib import Path
import numpy as np

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import (
        colors, save_one_box, check_requirements, increment_path, print_args, select_device, 
        LOGGER, Profile,
        loadData
    )
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from libs.st_gcn.StgcnApi import ActionEstimation as PhoneActionEstimation
from libs.st_gcn.TwoStreamStgcn import ActionEstimation as StandActionEstimation
from _utils.PointsUtils import pointsAnyInBox, xywh2xyxy
from _utils.PoseTransformer import getBodyPartIndex, toBoneboxCoord, coco2017Keypoints2CocoCut as co2cocut


class PhoneWalkDetector:

    def __init__(self, device):
        device = select_device(device)
        self.__loadModels(device)


    def __loadModels(self, device):
        # people detector
        self.peoDt =  TargetsDetector(
            weights='libs/yolov5/weights/yolov5s.pt',
            data='libs/yolov5/data/coco128.yaml',
            device=device
        )
        # phone detector
        self.phoneDt= TargetsDetector(
            weights='libs/yolov5/weights/phone_ep20.pt',
            data='libs/yolov5/data/phone.yaml',
            device=device
        )
        # people pose estimation
        self.poseEstimation = SingleImagePoseEstimation(device=device)
        # action estimation of holding phone with hand(s) 
        self.phoneAe = PhoneActionEstimation(
            weight_file='libs/st_gcn/model/stgcn_class3_150_94_ex9.pt',
            class_names=['nohand', 'oneHand', 'twoHands'],
            device=device
        )
        # action estimation of sitting and standing
        self.walkAe = StandActionEstimation(device=device)


    def phoneWalkingAeOfSingleImage(self, keypoints, score, dt : Profile):
        '''
            phoneWalking Action Astimation Of Single Image
            params:
                keypoints: not normalizied skeleton keypoints
                score : confidence of keypoints
        '''
        kp = toBoneboxCoord(co2cocut(keypoints, [17, 2]), norm=True) # normalizied keypoints according to skeletion box
        walkingActionEstimation = self.walkAe
        sit, sitTime = walkingActionEstimation.predictSingleCap(kp, co2cocut(score, [17,1]), None, normed=True)
        dt.t += sitTime
        sit = walkingActionEstimation.getLabel(sit)  
        if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
            return False
        
        kp = toBoneboxCoord(keypoints, norm=True)
        phoneActionEstimation = self.phoneAe
        phone, phoneTime = phoneActionEstimation.predictSingleCap(kp, score, None, normed=True)
        dt.t += phoneTime
        phone = phoneActionEstimation.getLabel(phone)
        if phone == 'nohand':
            return False
        
        return True

    def phoneWalkingAeOfMultiCaps(self, tvc, dt : Profile):
        '''
            phoneWalking Action Astimation Of Multi Caps
            params:
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
        walkingActionEstimation = self.walkAe
        sit, sitTime = walkingActionEstimation.predict(np.array(cococutTvc), None, normed=True)
        dt.t += sitTime
        sit = walkingActionEstimation.getLabel(sit) 
        if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
            return False
        
        for i,vc in enumerate(tvc):
            tvc[i][:,:2] = toBoneboxCoord(vc[:,:2], norm=True)
        phoneActionEstimation = self.phoneAe
        phone, phoneTime = phoneActionEstimation.predict(tvc, None, normed=True)
        phone = phoneActionEstimation.getLabel(phone)
        dt.t += phoneTime
        if phone == 'nohand':
            return False
        
        return True, sitTime + phoneTime

    @staticmethod
    def phoneInHand(keypoints, poseFormat, phoneXyxy):
        wristpoints = keypoints[getBodyPartIndex(poseFormat, 'wrist')]
        if pointsAnyInBox(wristpoints, phoneXyxy, 0.75):
            return True
        return False

    @staticmethod
    def phoneInEars(keypoints, poseFormat, phoneXyxy):
        earpoints = keypoints[getBodyPartIndex(poseFormat, 'ear')]
        if pointsAnyInBox(earpoints, phoneXyxy, 0):
            return True
        return False


    def playPhoneDetection(self, peoCrop, cropBox, peoKeypoints, poseType, dt):
        '''
            param:
                cropBox: people crop coord based img box
            return:
                phone box based img box
        '''
        # (手持)手机检测
        _, phoneXyxyBoxes,_, time = self.phoneDt.detectSingleImage(peoCrop, classes=[0], conf_thres=0.4)
        dt.t += time
        pB = None
        action = -1
        if(len(phoneXyxyBoxes)):   # 人像图中存在手机
            for phoneBox in phoneXyxyBoxes:  # 对于每个手机，是否与人手重合
                phoneBox += np.array(cropBox)[[0, 1, 0, 1]]
                if self.phoneInHand(peoKeypoints, poseType, phoneBox):
                    if self.phoneInEars(peoKeypoints, poseType, phoneBox):
                        pB = phoneBox
                        action = 1
                    else:
                        return phoneBox, 0
        return pB, action


    def drawPlayphoneAndCall(self, annotator, peopleBox, phoneBox, actionId):
        red_bgr = (0, 0, 256)
        black_bgr = (0, 0, 0)
        color = red_bgr if actionId == 0 else colors(0)
        annotator.box_label(peopleBox, label=self.getLabel(actionId), color=color)   # 深拷贝，会直接在原始图片上进行修改
        annotator.box_label(phoneBox, label='phone', color=colors(5), txt_color=black_bgr)


    def detectSingleImage(self, im0, mode, isNew, line_thickness = 2):

        targetBoxes = []
        crops = []
        labelIdxs = []

        def fillResults(crop, peopleBox, phoneBox, actionId):
            targetBoxes.append(peopleBox)
            labelIdxs.append(actionId)
            targetBoxes.append(phoneBox)
            labelIdxs.append(2)
            crops.append(crop)

        img = im0.copy()

        # 检测人像
        _, peopleXyxyBoxes, confs, time = self.peoDt.detectSingleImage(img, classes=[0])

        # time recorder
        dt = [Profile(), Profile()]
        peTime = 0.0
        
        if(len(peopleXyxyBoxes) > 0):   # 存在人像

            # annotator（drawer）
            annotator = TargetsAnnotator(img, line_thickness)

            if mode == 'image':

                # 根据人像检测骨骼结点
                poses, peTime = self.poseEstimation.process(img, peopleXyxyBoxes, confs) # 获得骨骼结点 

                # 对每个骨骼结点进行手机检测和动作检测
                for pose in poses: #   对于每个人像

                    peopleBox = xywh2xyxy(pose['bbox'])  # 获得此人的人像盒子xyxy
                    keypoints = pose['keypoints'] # 获得此人的骨骼结点
                    score = pose['kp_score'] # 获得此人的骨骼结点置信度

                    # action estimation
                    if not self.phoneWalkingAeOfSingleImage(keypoints, score, dt[0]):
                        continue

                    # (手持)手机检测
                    crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                    phoneBox, actionId = self.playPhoneDetection(crop, peopleBox, keypoints, self.poseEstimation.poseType, dt[1])
                    if phoneBox is not None:
                        self.drawPlayphoneAndCall(annotator, peopleBox, phoneBox, actionId)
                        fillResults(crop, peopleBox, phoneBox, actionId)

            else:

                if isNew: # new video or first video
                    self.poseEstimation.initTracker()
                    self.poseStore = []  # peos in last frame
                
                poses, peTime = self.poseEstimation.process(im0, peopleXyxyBoxes, confs, tracking=True)

                existedPeo = ([], {}) # peos in now frame
                poseStore = self.poseStore
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
                        if len(tvc) > 1:
                            if not self.phoneWalkingAeOfMultiCaps(tvc, dt[0]):
                                continue
                            peopleBox = existedPeo[1][id]
                            crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                            phoneBox, actionId = self.playPhoneDetection(crop, peopleBox, tvc[-1][:,:2], self.poseEstimation.poseType, dt[1])
                            if phoneBox is not None:
                                self.drawPlayphoneAndCall(annotator, peopleBox, phoneBox, actionId)
                                fillResults(crop, peopleBox, phoneBox, actionId)

                    else:
                        tvc.clear()

            # end of img/video-cap process --------------------------------------------------------------------------------

            img = annotator.result()
        
        # end of people detector -----------------------------------------------------------------------

        # cls, boxes, crops, annotatedImages, time,
        return labelIdxs, targetBoxes, crops, img, [time, peTime, dt[0].t, dt[1].t]

    def getLabel(self, cls):
        return ['phoneWalking', 'call', 'phone'][cls]
