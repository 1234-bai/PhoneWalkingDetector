import torch
import sys
import cv2
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Alphapose root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from alphapose.models import FastPose
from alphapose.utils.transforms import heatmap_to_coord_simple, get_affine_transform, im_to_torch
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.bbox import (_box_to_center_scale, _center_scale_to_box)
from alphapose.utils.vis import getTime
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tracker_cfg
from trackers import track


class AlphaposeDataTransformer():

    @staticmethod
    def imagePreprocess(
        src, 
        bbox,
        input_size, # h,w
    ):
        xmin, ymin, xmax, ymax = bbox
        inp_h, inp_w = input_size
        _aspect_ratio = float(inp_w) / inp_h
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, _aspect_ratio)
        scale = scale * 1.0


        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

    @staticmethod
    def heatmap2Pose(
        boxes,  # xyxy
        croppedBoxes,  # xywh
        scores,
        ids,
        hm,
        numJoints,
        normType,# 激活函数类型
        hmSize,
        useHeatmapLoss,
        heatmap2coord,
        min_box_area=0, # min box area to filter out
        tracking = False
    ):

        assert(boxes is not None and len(boxes) != 0)
        # location prediction (n, kp, 2) | score prediction (n, kp, 1)
        assert hm.dim() == 4
        eval_joints = list(range(numJoints))
        pose_coords = []
        pose_scores = []

        time = getTime()
        for i in range(hm.shape[0]):
            bbox = croppedBoxes[i].tolist()
            pose_coord, pose_score = heatmap2coord(hm[i][eval_joints], bbox, hm_shape=hmSize, norm_type=normType)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        if not tracking:
            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                    pose_nms(boxes, scores, ids, preds_img, preds_scores, min_box_area, use_heatmap_loss=useHeatmapLoss)
        _, time = getTime(time)
        
        _result = []
        for k in range(len(scores)):
            _result.append({
                'keypoints':preds_img[k],
                'kp_score':preds_scores[k],
                'proposal_score': (torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k])) / 3,
                'idx':ids[k],
                'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0], boxes[k][3]-boxes[k][1]]
            })

        return _result, time
    
    @staticmethod
    def viewpPoseInImage(
        image, # BGR
        poses,
        vis_threshold,  # 可视阈值
        vis_fast = False,
        showbox=False, 
        tracking=False
    ):

        assert(image is not None)

        if vis_fast:
            from alphapose.utils.vis import vis_frame_fast as vis_frame
        else:
            from alphapose.utils.vis import vis_frame
        opt = edict({
            'pose_track':False,
            'tracking':tracking,
            'showbox':showbox,  
        })
        return vis_frame(image, poses, opt, vis_threshold)
    


# process single single-human image
class SingleImagePoseEstimation():

    def __init__(
        self, 
        device : torch.device,
        checkpoint='weights/alphapose/fast_res50_256x192.pth',
    ):

        self.device = device
        cfg={
            'NUM_JOINTS': 17,
            'IMAGE_SIZE': [256, 192],
            'HEATMAP_SIZE': [64, 48],
            'NUM_DECONV_FILTERS': [256, 256, 256],
            'NUM_LAYERS': 50
        }

        # Load pose model
        self.pose_model = FastPose(**cfg)
        print(f'Loading pose model from {checkpoint}...')
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.pose_model.to(self.device)
        self.pose_model.eval()
       
        self.__num_joints = cfg['NUM_JOINTS']
        self.__image_size = cfg['IMAGE_SIZE']
        self.__heatmap_size = cfg['HEATMAP_SIZE']
        # load image preprocess transormer
        self.transformation = AlphaposeDataTransformer.imagePreprocess
        # load profile of pose visualize profile
        self.__vis_thres = [0.4] * self.__num_joints

        # self.tracker = Tracker(tcfg, self.device)

    def process(
        self, 
        image, # HWC, BGR
        boxes, 
        confs,
        tracking = False
    ):
        '''
            return list of 'keypoints:list , scores:list, box: list of 4' which index is people_number
        '''
        with torch.no_grad():
            assert(image is not None)

            # pre process cropped human image for pose estimation
            image = np.array(image, dtype=np.uint8)[:, :, ::-1] # image channel BGR->RGB
            inps = torch.zeros(len(boxes), 3, *(self.__image_size))
            cropped_boxes = []
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation(image, box, self.__image_size) # box is xyxy, cropped_box is xywh from box
                # cropped_boxes = torch.FloatTensor([cropped_box])
                cropped_boxes.append(cropped_box)

            time = getTime()

            # Pose Estimation
            inps = inps.to(self.device)
            hm = self.pose_model(inps)  # heatmap

            # tracking
            if tracking:
                boxes,confs,ids,hm,cropped_boxes = track(self.tracker,self.device,inps,boxes,hm,cropped_boxes,confs)

            _, time = getTime(time)

            # transform heatmap data to pose data
            poses, postProcessTime = AlphaposeDataTransformer.heatmap2Pose(
                torch.FloatTensor(boxes), 
                torch.FloatTensor(cropped_boxes), 
                torch.FloatTensor(confs), 
                ids if tracking  else torch.Tensor(range(len(confs))), 
                hm,
                self.__num_joints,
                None,
                self.__heatmap_size,
                True,
                heatmap_to_coord_simple,
                tracking = tracking
            )
            
        return poses, time + postProcessTime
    
    @property
    def vis_thres(self):
        return self.__vis_thres
    
    @property
    def poseType(self):
        return 'Mscoco'


    def initTracker(self):
        self.tracker = Tracker(tracker_cfg, self.device)
