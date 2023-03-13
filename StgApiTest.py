import numpy as np
import cv2

from PointsUtils import pt2bbox
from libs.Track.Tracker import Tracker, Detection
from libs.yolov5.yolov5DetectorApi import TargetsDecetor, TargetsAnnotator
from libs.yolov5.utils.plots import colors
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer
from libs.st_gcn.StgcnApi import ActionEstimation


ae = ActionEstimation(weight_file='libs\st_gcn\model\st-gcn-tsstg-fail-model.pth')

poseTest = SingleImagePoseEstimation(
    configFilePath='libs\\Alphapose\\configs\\coco\\resnet\\256x192_res50_lr1e-3_1x.yaml',
    checkpoint='libs\\Alphapose\\pretrained_models\\fast_res50_256x192.pth',
    device=0
)

test =  TargetsDecetor(
    weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\yolov5s.pt',
    data='libs\\yolov5\\data\\coco128.yaml'
)
dataset = test.loadData(source='data/images')

tracker = None
preFilename = ''
for path, _, im0s, vid_cap, s in dataset:
    if dataset.mode == 'image': continue
    # 获得文件名字
    filename = path[0] if dataset.mode == 'stream' else path

    # 获得原始图片
    im0 = im0s[0] if dataset.mode == 'stream' else im0s

    # 注释器（画图器）
    annotator = TargetsAnnotator(im0, 2)

    # 检测人像
    _, peopleXyxyBoxes, crops, confs = test.detectorSingleImg(im0, classes=[0], conf_thres=0.4)
    if(len(peopleXyxyBoxes) > 0):

        if dataset.mode == 'image':
            # 根据人像检测骨骼结点
            poses = poseTest.process(im0, peopleXyxyBoxes, confs) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number

            for i,crop in enumerate(crops): #   对于每个人像
                keypoints = poses[i]['keypoints'] # 获得此人的骨骼结点
                scores = poses[i]['kp_score'] # 获得此人的骨骼结点置信度
                
                # 动作检测
                # 骨骼结点格式转换，与动作检测模型的骨骼结点输入格式匹配
                keypoints = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(keypoints, [17, 2])
                scores = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(scores, [17, 1])
                box = poses[i]['bbox'] # 每个人像的xywhBox
                actionName = ae.predictSingleCap(keypoints, scores, (box[2], box[3]))
                
                annotator.box_label(peopleXyxyBoxes[i], label=actionName, color=colors(0))
        else:   # stream or vedio for tracker
            if preFilename != filename: # 新的视频或者第一个视频
                # 旧的Tracker在tracker不指向它的时候，被Python垃圾回收机制自动回收
                tracker = Tracker(n_init=3)
                preFilename = filename
            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks:
                det = track.to_tlbr().tolist()
                peopleXyxyBoxes.append(det)
                confs.append(0.5)
            # 根据人像检测骨骼结点
            poses = poseTest.process(im0, peopleXyxyBoxes, confs)
            # Create Detections object.
            detections = []
            for ps in poses:
                kp = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(ps['keypoints'], [17, 2])
                sc = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(ps['kp_score'], [17, 1])
                detections.append(
                    Detection(
                        pt2bbox(kp),
                        np.concatenate((kp,sc), axis=1),
                        sc.mean()
                    )
                )
            tracker.update(detections)
            # Predict Actions of each track.
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue
                bbox = track.to_tlbr().astype(int)
                actionName = 'pending..'
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) >= 10:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    actionName = ae.predict(pts, im0.shape[:2])
                if actionName == 'Walking':
                    annotator.box_label(bbox, actionName)


    img = annotator.result()
    cv2.imshow(str(path), img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


