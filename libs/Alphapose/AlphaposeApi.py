import torch
import os
import platform
import sys
import math
import time
import cv2
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Alphapose root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.vis import getTime
from libs.yolov5.utils.torch_utils import select_device
from PointsUtils import twoPointsSuperpose


class AlphaposeDataTransformer():

    @staticmethod
    def heatmap2Pose(
        image, # BGR
        boxes,  # xyxy
        cropped_boxes,  # xywh
        scores,
        ids,
        hm,
        numJoints,
        norm_type,
        hm_size,
        use_heatmap_loss,
        heatmap_to_coord,
        min_box_area=0, # min box area to filter out
    ):

        assert(image is not None and len(image) != 0)
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

        boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, min_box_area, use_heatmap_loss=use_heatmap_loss)
        
        _result = []
        for k in range(len(scores)):
            _result.append({
                'keypoints':preds_img[k],
                'kp_score':preds_scores[k],
                'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                'idx':ids[k],
                'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
            })

        return _result
    
    @staticmethod
    def viewpPoseInImage(
        image, # BGR
        pose,
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
        return vis_frame(image, pose, opt, vis_threshold)
        
    @staticmethod
    def writeJson(
        pose, 
        outputpath, 
        form='coco', 
        for_eval=False
    ):
        from alphapose.utils.pPose_nms import write_json
        write_json(pose, outputpath, form=form, for_eval=for_eval)
        print("Results have been written to json.")


    # coco2017format skeleton to openposeCocoFormat skeleton
    @staticmethod
    def coco2017Keypoints2openposeCoco(coco2017, inputSize=[17, 3]):
        # refer to : 
        # https://github.com/jin-s13/COCO-WholeBody, 
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#body-keypoint-ordering-in-c-python
        coco2017 = torch.FloatTensor(coco2017)
        res = torch.zeros(18, *(inputSize[1::]))
        res[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] = \
            coco2017[[0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
        res[1] = (coco2017[5] + coco2017[6])/2.0
        return res.numpy()
    
    @staticmethod
    def coco2017Keypoints2CocoCut(coco2017, inputSize=[17, 3]):
        coco2017 = torch.FloatTensor(coco2017)
        res = torch.zeros(14, *(inputSize[1::]))
        res[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] = \
            coco2017[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        res[13] = (coco2017[5] + coco2017[6])/2.0
        return res.numpy()
    


# process single single-human image
class SingleImagePoseEstimation():

    def __init__(self, configFilePath, checkpoint, device):

        cfg = update_config(configFilePath)
        self.cfg = cfg
        # device = torch.device("cuda:" + str(device) if device >= 0 else "cpu")
        self.device = select_device(device)
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

    def process(
        self, 
        image, # BGR
        boxes, 
        confs, 
        flipFlag=False
    ):
        with torch.no_grad():
            assert(image is not None)
            # pre process cropped human image for pose estimation
            image = np.array(image, dtype=np.uint8)[:, :, ::-1] # image channel BGR->RGB
            inps = torch.zeros(len(boxes), 3, *(self.transformation._input_size))
            cropped_boxes = []
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(image, box)
                # cropped_boxes = torch.FloatTensor([cropped_box])
                cropped_boxes.append(cropped_box)
            # Pose Estimation
            inps = inps.to(self.device)
            if flipFlag:
                inps = torch.cat((inps, flip(inps)))
            hm = self.pose_model(inps)
            if flipFlag:
                hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                hm = hm.cpu()
            # transform heatmap data to pose data
            poses = AlphaposeDataTransformer.heatmap2Pose(
                image,
                torch.FloatTensor(boxes), 
                torch.FloatTensor(cropped_boxes), 
                torch.FloatTensor(confs), 
                torch.Tensor(torch.zeros(len(confs))), 
                hm,
                self.cfg.DATA_PRESET.NUM_JOINTS,
                self.cfg.LOSS.get('NORM_TYPE', None),
                self.cfg.DATA_PRESET.HEATMAP_SIZE,
                self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss',
                get_func_heatmap_to_coord(self.cfg)
            )
        return poses

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
        self.__vis_thres__ = vis_thres

    def getVisThres(self):
        return self.__vis_thres__

    def __setTransformation(self):
         # load image preprocess transormer
        cfg = self.cfg
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self.pose_dataset, 
                scale_factor=0,
                input_size=cfg.DATA_PRESET.IMAGE_SIZE,
                output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
                rot=0, sigma=cfg.DATA_PRESET.SIGMA,
                train=False, 
                add_dpg=False, 
                gpu_device=self.device)
        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2)
            })
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set, 
                scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2,2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR, 
                sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, 
                add_dpg=False, 
                gpu_device=self.device,
                loss_type=cfg.LOSS['TYPE']
            )


    def getHandIndex(self):
        if(self.poseType == 'Mscoco'):
            return [10, 11]
        elif(self.poseType == 'Halpe_26'):
            return [9, 10]
        return [10, 11]
    
    def getEarIndex(self):
        if(self.poseType == 'Mscoco'):
            return [3, 4]
        elif(self.poseType == 'Halpe_26'):
            return [3, 4]
        return [3, 4]
