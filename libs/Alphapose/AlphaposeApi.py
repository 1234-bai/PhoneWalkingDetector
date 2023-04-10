import torch
import sys
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Alphapose root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.vis import getTime
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track


class AlphaposeDataTransformer():

    @staticmethod
    def heatmap2Pose(
        boxes,  # xyxy
        cropped_boxes,  # xywh
        scores,
        ids,
        hm,
        numJoints,
        norm_type,# 激活函数类型
        hm_size,
        use_heatmap_loss,
        heatmap_to_coord,
        min_box_area=0, # min box area to filter out
        tracking = False
    ):

        assert(boxes is not None and len(boxes) != 0)
        # location prediction (n, kp, 2) | score prediction (n, kp, 1)
        assert hm.dim() == 4
        if hm.size()[1] == 136:
            eval_joints = [*range(0,136)]
        elif hm.size()[1] == 26:
            eval_joints = [*range(0,26)]
        elif hm.size()[1] == 133:
            eval_joints = [*range(0,133)]
        else:
            eval_joints = list(range(numJoints))
        pose_coords = []
        pose_scores = []

        time = getTime()
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            if isinstance(heatmap_to_coord, list):
                pose_coords_body_foot, pose_scores_body_foot = heatmap_to_coord[0](
                    hm[i][eval_joints[:-110]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords_face_hand, pose_scores_face_hand = heatmap_to_coord[1](
                    hm[i][eval_joints[-110:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = heatmap_to_coord(hm[i][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        if not tracking:
            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                    pose_nms(boxes, scores, ids, preds_img, preds_scores, min_box_area, use_heatmap_loss=use_heatmap_loss)
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
        configFilePath='libs/Alphapose/configs/coco_256x192_res50_lr1e-3_1x.yaml',
        checkpoint='weights/alphapose/fast_res50_256x192.pth',
    ):

        cfg = update_config(configFilePath)
        self.cfg = cfg
        self.device = device
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        self.poseType = cfg.DATASET.TRAIN.TYPE

        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print(f'Loading pose model from {checkpoint}...')
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.pose_model.to(self.device)
        self.pose_model.eval()
       
        self.__setTransformation()
        self.__setVisThres()

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
            inps = torch.zeros(len(boxes), 3, *(self.transformation._input_size))
            cropped_boxes = []
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(image, box) # box is xyxy, cropped_box is xywh from box
                # cropped_boxes = torch.FloatTensor([cropped_box])
                cropped_boxes.append(cropped_box)

            time = getTime()

            # Pose Estimation
            inps = inps.to(self.device)
            hm = self.pose_model(inps)

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
                self.cfg.DATA_PRESET.NUM_JOINTS,
                self.cfg.LOSS.get('NORM_TYPE', None),
                self.cfg.DATA_PRESET.HEATMAP_SIZE,
                self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss',
                get_func_heatmap_to_coord(self.cfg),
                tracking = tracking
            )
            
        return poses, time + postProcessTime

    def __setVisThres(self):
        # load profile of pose visualize profile
        loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        vis_thres = [0.4] * num_joints
        if loss_type != 'MSELoss':
            if 'JointRegression' in loss_type:
                vis_thres = [0.05] * num_joints
            elif loss_type == 'Combined':
                if num_joints == 68:
                    hand_face_num = 42
                else:
                    hand_face_num = 110
                vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num
        self.__vis_thres = vis_thres

    def getVisThres(self):
        return self.__vis_thres

    def __setTransformation(self):
         # load image preprocess transormer
        cfg = self.cfg
        self.transformation = SimpleTransform(
            self.pose_dataset, 
            scale_factor=0,
            input_size=cfg.DATA_PRESET.IMAGE_SIZE,
            output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
            rot=0, sigma=cfg.DATA_PRESET.SIGMA,
            train=False, 
            add_dpg=False, 
            gpu_device=self.device
        )

    def initTracker(self):
        self.tracker = Tracker(tcfg, self.device)
