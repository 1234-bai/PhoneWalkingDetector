import numpy as np

from libs.yolov5.yolov5DetectorApi import TargetsDetector, TargetsAnnotator
from libs.yolov5 import colors, save_one_box, select_device,Profile
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation
from actionModels.StgcnAll import AllStgcn
from actionModels.StgcnUpDown import UpDownStgcn
from utils.PointsUtils import xywh2xyxy

from Detector import Detector


class PhoneActionEstimation(Detector):

    def __init__(self, device = '', model_type = 1):
        super().__init__(['walk', 'oneHand', 'twoHands', 'other'])
        device = select_device(device)
        self.__loadModels(device, model_type)


    def __loadModels(self, device, model_type):
        # people detector
        self.peoDt =  TargetsDetector(
            weights='weights/yolov5/yolov5s.pt',
            device=device
        )
        # people pose estimation
        self.poseEstimation = SingleImagePoseEstimation(device=device)
        # action estimation of holding phone with hand(s)
        models = [AllStgcn, UpDownStgcn]
        self.phoneAe = models[model_type-1](device=device)


    def drawPlayphoneAndCall(self, annotator, peopleBox, actionId, conf):
        red_bgr = (0, 0, 256)
        color = red_bgr if actionId == 1 or actionId == 2 else colors(0)
        annotator.box_label(peopleBox, label=self.getLabel(actionId)+f':{conf:.2f}', color=color)   # 深拷贝，会直接在原始图片上进行修改


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
                    pwConf = self.phoneAe.predictSingleCap(keypoints, score, dt[0]) # confidenece of phonewalking estimation   
                    crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                    actionId = np.array(pwConf).argmax()
                    if pwConf[actionId] >= conf_thres:
                        self.drawPlayphoneAndCall(annotator, peopleBox, actionId, pwConf[actionId])
                        fillResults(crop, peopleBox, actionId, pwConf[actionId])
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
                            pwConf =  self.phoneAe.predictMultiCaps(tvc, dt[0])
                            # get crop
                            peopleBox = existedPeo[1][id][0]
                            crop = save_one_box(peopleBox, im0, save=False, BGR=True)
                            actionId = np.array(pwConf).argmax()
                            if pwConf[actionId] >= conf_thres:
                                self.drawPlayphoneAndCall(annotator, peopleBox, actionId, pwConf[actionId])
                                fillResults(crop, peopleBox, actionId, pwConf[actionId])
                    else:
                        tvc.clear()

            # end of img/video-cap process --------------------------------------------------------------------------------

            img = annotator.result()
        
        # end of people detector -----------------------------------------------------------------------

        # cls, boxes, confs, crops, annotatedImage, time,
        return labelIds, targetBoxes, confs, crops, img, [time, peTime, dt[0].t, dt[1].t]

