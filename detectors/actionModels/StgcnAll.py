from libs.yolov5 import Profile
from libs.st_gcn.StgcnApi import ActionEstimation
from utils.PoseTransformer import toBoneboxCoord

class AllStgcn:
    def __init__(self, device):
        self.ae = ActionEstimation(
            weight_file='weights/stgcn/action_stgcn_all_ep500.pt',
            class_names=['walk', 'onehand', 'twohand', 'other'],
            device=device
        )

    def predictSingleCap(self, keypoints, score, dt : Profile):
        kp = toBoneboxCoord(keypoints, norm=True) - 0.5    # normalization and centralization
        conf, time = self.ae.predictSingleCap(kp, score, None, normed=True) 
        dt.t +=time
        # conf: (walking, play with one, play with two, other)
        return conf
    
    def predictMultiCaps(self, tvc, dt : Profile):
        for i,vc in enumerate(tvc):
            tvc[i][:,:2] = toBoneboxCoord(vc[:,:2], norm=True) - 0.5    # normalization and centralization
        conf, time = self.ae.predict(tvc, None, normed=True)   
        dt.t += time
        # conf: (walk, play with one, play with two, other)
        
        return conf
