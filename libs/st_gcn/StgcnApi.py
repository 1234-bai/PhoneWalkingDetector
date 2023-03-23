import os
import torch
import numpy as np

from .net.st_gcn import Model


class ActionEstimation():
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        weight_file = 'libs/st_gcn/model/stgcn_class7.pt',
        class_names = ['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'Photograph', 'Stand', 'Sit', 'Other'],
        layout='Mscoco',
        device='cuda'
    ):
        graph_args = {
            'strategy': 'spatial',
            'layout': layout,
        }
        in_channels = 3
        self.count_class = len(class_names)
        edge_importance_weighting = True
        self.device = device
        self.class_names = class_names

        self.model = Model(in_channels, self.count_class, graph_args, edge_importance_weighting).to(device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()
        
    
    def predictSingleCap(self, keypoints, kp_scores, image_size, normed = False, copy_times=30):
        """Predict actions from single person skeleton points and score in a cap.
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

        pts = [np.concatenate((keypoints, kp_scores), axis=1)] * copy_times     # 骨骼和置信度结合
        return self.predict(np.array(pts), image_size, normed)
    
    def predict(self, pts, image_size, normed=False):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        if not normed:
            pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :, :, :, None] # N,C,T,V,M

        pts = pts.to(self.device)

        out = self.model(pts)
        out = out.detach().cpu().numpy()
        print(out)
        return out[0].argmax()


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
