import os
import torch
import numpy as np

from .net.tssg import TwoStreamSpatialTemporalGraph as Model


class ActionEstimation():
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """

    def __init__(self,
        weight_file,
        class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down', 'Fall Down'],
        device='cuda'
    ):
        graph_args = {
            'strategy': 'spatial',
            # 'layout': 'openpose'
            'layout' : 'coco_cut'
        }
        # num_class = 400
        # in_channels = 3
        self.count_class = len(class_names)
        edge_importance_weighting = True
        self.device = device
        self.class_names = class_names

        # self.model = Model(in_channels, num_class, graph_args, edge_importance_weighting).to(device)
        self.model = Model(graph_args, self.count_class, edge_importance_weighting).to(device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()
        
    
    def predict(self, keypoints, kp_scores, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            keypoints: (numpy) points in shape `(v, c)` where
                v : number of graph node (body parts).,
                c : channel (x, y).,
            kp_score: (numpy) score in shape `(v, c)` where
                v : number of graph node (body parts).,
                c : channel (score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (str) action name.
        """
        pts = np.concatenate((keypoints, kp_scores), axis=1)    # 骨骼和置信度结合

        pts[:, :2] = normalize_points_with_size(pts[:, :2], image_size[0], image_size[1])
        pts[:, :2] = scale_pose(pts[:, :2])
        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(1, 0)[None, :, None, :] # N, C, T, V
        pts = pts.to(self.device)
        mot = pts[:, :2, :, :]


        out = self.model((pts, mot))
        out = out.detach().cpu().numpy()
        return self.class_names[out[0].argmax()]

    def getLabel(self, num_class):
        assert(num_class >= 0 and num_class < self.count_class)
        return self.class_names[num_class]

def normalize_points_with_size(xy, width, height, flip=False):
    """Normalize scale points in image with size of image to (0-1).
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    if flip:
        xy[:, :, 0] = 1 - xy[:, :, 0]
    return xy


def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()