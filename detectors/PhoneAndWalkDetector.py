import numpy as np

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, save_one_box, select_device,Profile
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from libs.st_gcn.StgcnApi import ActionEstimation as PhoneActionEstimation
from libs.st_gcn.TwoStreamStgcn import ActionEstimation as StandActionEstimation
from _utils.PointsUtils import pointsAnyInBox, xywh2xyxy
from _utils.PoseTransformer import getBodyPartIndex, toBoneboxCoord, coco2017Keypoints2CocoCut as co2cocut

from Detector import Detector


class PhoneWalkDetector(Detector):

    def __init__(self, device = ''):
        super().__init__(['call', 'playWithOneHand', 'playWithTwoHands', 'walking', 'other'])
        device = select_device(device)
        self.__loadModels(device)


    def __loadModels(self, device):
        # people detector
        self.peoDt =  TargetsDetector(
            weights='weights/yolov5/yolov5s.pt',
            device=device
        )
        # phone detector
        self.phoneDt= TargetsDetector(
            weights='weights/yolov5/phoneEx5.pt',
            device=device
        )
        # people pose estimation
        self.poseEstimation = SingleImagePoseEstimation(device=device)
        # action estimation of holding phone with hand(s) 
        self.phoneAe = PhoneActionEstimation(
            weight_file='weights/stgcn/stgcn_class3_150_94_ex9.pt',
            class_names=['nohand', 'oneHand', 'twoHands'],
            device=device
        )
        # action estimation of sitting and standing
        self.walkAe = StandActionEstimation(device=device)


    def __phoneWalkingAeOfSingleImage(self, keypoints, score, dt : Profile):
        '''
            phoneWalking Action Astimation Of Single Image
            params:
                keypoints: not normalizied skeleton keypoints
                score : confidence of keypoints
        '''
        kp = toBoneboxCoord(co2cocut(keypoints, [17, 2]), norm=True) # normalizied keypoints according to skeletion box
        walkingActionEstimation = self.walkAe
        out, time = walkingActionEstimation.predictSingleCap(kp, co2cocut(score, [17,1]), None, normed=True)
        dt.t += time
        conf = [out[[0, 1, 4]].sum(), out[[2, 3, 5, 6]].sum()]  # conf: (walk, not walk)  
        # sit = walkingActionEstimation.getLabel(out)
        # if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
        #     return False, conf
        
        kp = toBoneboxCoord(keypoints, norm=True) - 0.5    # normalization and centralization
        phoneActionEstimation = self.phoneAe
        out, time = phoneActionEstimation.predictSingleCap(kp, score, None, normed=True) 
        dt.t +=time
        conf = [*(conf[0] * out), conf[1]]    # conf: (walking, play with one, play with two, other)
        # phone = phoneActionEstimation.getLabel(phone)
        # if phone == 'nohand':
        #     return False
        
        return conf

    def __phoneWalkingAeOfMultiCaps(self, tvc, dt : Profile):
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
        out, time = walkingActionEstimation.predict(np.array(cococutTvc), None, normed=True)
        dt.t += time
        conf = [out[1], 1 - out[1]]  # conf: (walk, not walk) 
        # sit = walkingActionEstimation.getLabel(sit) 
        # if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
        #     return False
        
        for i,vc in enumerate(tvc):
            tvc[i][:,:2] = toBoneboxCoord(vc[:,:2], norm=True) - 0.5    # normalization and centralization
        phoneActionEstimation = self.phoneAe
        out, time = phoneActionEstimation.predict(tvc, None, normed=True)   
        dt.t += time
        conf = [*(conf[0] * out), conf[1]]    # conf: (walk, play with one, play with two, other)
        # phone = phoneActionEstimation.getLabel(phone)
        # if phone == 'nohand':
        #     return False
        
        return conf

    @staticmethod
    def phoneInHand(keypoints, poseFormat, phoneXyxy):
        wristpoints = keypoints[getBodyPartIndex(poseFormat, 'wrist')]
        if pointsAnyInBox(wristpoints, phoneXyxy, 0.75):
            return True
        return False

    @staticmethod
    def phoneInEars(keypoints, poseFormat, phoneXyxy):
        earpoints = keypoints[getBodyPartIndex(poseFormat, 'ear')]
        if pointsAnyInBox(earpoints, phoneXyxy, 0.45):
            return True
        return False


    def __playPhoneDetection(self, peoCrop, cropBox, peoKeypoints, poseType, dt):
        '''
            param:
                cropBox: people crop coord based img box
            return:
                phone box based img box
                actionId based phone 
        '''
        # (手持)手机检测
        _, phoneXyxyBoxes, confs, time = self.phoneDt.detectSingleImage(peoCrop, conf_thres=0.6, classes=[0])
        dt.t += time
        pB = None
        conf = [0.0, 0.0, 1.0]   # (phone in hand, phone ears, other)
        if(len(phoneXyxyBoxes)):   # 人像图中存在手机
            for i, phoneBox in enumerate(phoneXyxyBoxes):  # 对于每个手机，是否与人手重合
                phoneBox += np.array(cropBox)[[0, 1, 0, 1]]
                if self.phoneInHand(peoKeypoints, poseType, phoneBox):
                    if self.phoneInEars(peoKeypoints, poseType, phoneBox):
                        pB = phoneBox
                        # conf[1:] = [confs[i], 1-confs[i]]
                        conf[1:] = [1.0, 0]
                    else:
                        # return phoneBox, [confs[i], 0.0, 1-confs[i]]
                        return phoneBox, [1.0, 0.0, 0.0]
        return pB, conf


    def __inferConfdience(
            self, 
            peoplePoseConf, 
            phoneWalkingConf, # (no, one, two, other)
            phoneConf   # (phone in hand, phone in ears, other)
        ):
        pwConf = peoplePoseConf * np.array(phoneWalkingConf)
        conf = np.concatenate((pwConf[:3], [1 - peoplePoseConf + pwConf[3]]))   # (walk, one, two, other)
        # total confidence
        conf = [
            conf[1] * phoneConf[1], # call = one * phone in ear
            conf[1] * phoneConf[0], # playWithOneHand = one * phone in hand
            conf[2] * (phoneConf[0] + phoneConf[1]), # playWithTwoHands = two * (phone in hand + phone in ear)
            conf[0] + (conf[1] + conf[2]) * phoneConf[2] # walk = walk + one * other + two * other
        ]   # (call, playWithOneHand, playWithTwoHands, walk)
        conf = [*conf, 1.0-sum(conf)]   # (call, playWithOneHand, playWithTwoHands, walk, other)
        return conf


    def drawPlayphoneAndCall(self, annotator, peopleBox, phoneBox, actionId, conf):
        red_bgr = (0, 0, 256)
        black_bgr = (0, 0, 0)
        color = red_bgr if actionId == 1 or actionId == 2 else colors(0)
        annotator.box_label(peopleBox, label=self.getLabel(actionId)+f':{conf:.2f}', color=color)   # 深拷贝，会直接在原始图片上进行修改
        if phoneBox is not None:
            annotator.box_label(phoneBox, label='phone', color=colors(5), txt_color=black_bgr)


    def detectSingleImage(self, im0, conf_thres, mode = 'image', isNew = True, line_thickness = 2):

        targetBoxes = []
        crops = []
        labelIds = []
        confs = []

        def fillResults(crop, peopleBox, actionId, conf):
            targetBoxes.append(peopleBox)
            labelIds.append(actionId)
            crops.append(crop)
            confs.append(conf)

        img = im0.copy()

        # 检测人像
        _, peopleXyxyBoxes, peoConfs, time = self.peoDt.detectSingleImage(img, classes=[0])

        # time recorder
        dt = [Profile(), Profile()]
        peTime = 0.0
        
        if(len(peopleXyxyBoxes) > 0):   # 存在人像

            # annotator（drawer）
            annotator = TargetsAnnotator(img, line_thickness)

            if mode == 'image':

                # 根据人像检测骨骼结点
                poses, peTime = self.poseEstimation.process(im0, peopleXyxyBoxes, peoConfs) # 获得骨骼结点 

                # 对每个骨骼结点进行手机检测和动作检测
                for pose in poses: #   对于每个人像

                    # action estimation
                    peopleBox = xywh2xyxy(pose['bbox'])  # 获得此人的人像盒子xyxy
                    keypoints = pose['keypoints'] # 获得此人的骨骼结点
                    score = pose['kp_score'] # 获得此人的骨骼结点置信度
                    boxConf = float(pose['proposal_score'].cpu())   # poeple pose conf
                    pwConf = self.__phoneWalkingAeOfSingleImage(keypoints, score, dt[0]) # confidenece of phonewalking estimation   

                    # (hold) phone detection
                    crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                    phoneBox, phoneConf = self.__playPhoneDetection(crop, peopleBox, keypoints, self.poseEstimation.poseType, dt[1])

                    conf = self.__inferConfdience(boxConf, pwConf, phoneConf)
                    actionId = np.array(conf).argmax()
                    if conf[actionId] >= conf_thres:
                        self.drawPlayphoneAndCall(annotator, peopleBox, phoneBox, actionId, conf[actionId])
                        fillResults(crop, peopleBox, actionId, conf[actionId])
            else:

                if isNew: # new video or first video
                    self.poseEstimation.initTracker()
                    self.poseStore = []  # peos in last frame
                
                poses, peTime = self.poseEstimation.process(im0, peopleXyxyBoxes, peoConfs, tracking=True)

                existedPeo = ([], {}) # peos in now frame
                poseStore = self.poseStore
                for ps in poses:
                    id = ps['idx']
                    existedPeo[0].append(id)
                    existedPeo[1][id] = (xywh2xyxy(ps['bbox']), float(ps['proposal_score'].cpu()))
                    kp = ps['keypoints']
                    vc = np.concatenate((kp,ps['kp_score']), axis=1)
                    try:
                        poseStore[id].append(vc)
                    except IndexError:
                        poseStore.append([vc])

                for id,tvc in enumerate(poseStore):
                    if id in existedPeo[0]:
                        if len(tvc) > 1:
                            # action estimation
                            pwConf =  self.__phoneWalkingAeOfMultiCaps(tvc, dt[0])
                            # phone detection
                            peopleBox = existedPeo[1][id][0]
                            crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                            phoneBox, phoneConf = self.__playPhoneDetection(crop, peopleBox, tvc[-1][:,:2], self.poseEstimation.poseType, dt[1])
                            # infer confidence 
                            conf = self.__inferConfdience(existedPeo[1][id][1], pwConf, phoneConf)
                            actionId = np.array(conf).argmax()
                            if conf[actionId] >= conf_thres:
                                self.drawPlayphoneAndCall(annotator, peopleBox, phoneBox, actionId, conf[actionId])
                                fillResults(crop, peopleBox, actionId, conf[actionId])
                    else:
                        tvc.clear()

            # end of img/video-cap process --------------------------------------------------------------------------------

            img = annotator.result()
        
        # end of people detector -----------------------------------------------------------------------

        # cls, boxes, crops, annotatedImages, time,
        return labelIds, targetBoxes, confs, crops, img, [time, peTime, dt[0].t, dt[1].t]

    # def getLabel(self, cls):
    #     # return ['phoneWalking', 'call', 'other'][cls]
    #     return self._Detector__class_names[cls]
