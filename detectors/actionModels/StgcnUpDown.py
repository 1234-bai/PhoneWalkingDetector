import numpy as np

from libs.yolov5 import Profile
from libs.st_gcn.StgcnApi import ActionEstimation as HandActionEstimation
# from libs.st_gcn.TwoStreamStgcn import ActionEstimation as StandActionEstimation
from utils.PoseTransformer import toBoneboxCoord, coco2017Keypoints2CocoCut as co2cocut

class UpDownStgcn:

    def __init__(self, device):
        # action estimation of holding phone with hand(s) 
        self.phoneAe = HandActionEstimation(
            weight_file='weights/stgcn/stgcn_class3_150_94_ex9.pt',
            class_names=['nohand', 'oneHand', 'twoHands'],
            device=device
        )
        # action estimation of sitting and standing
        # self.walkAe = StandActionEstimation(device=device)
        self.walkAe = HandActionEstimation(
            weight_file='weights/stgcn/action_stgcn_all_ep300.pt',
            class_names=['walk', 'onehand', 'twohand', 'other'],
            device=device
        )

    def predictSingleCap(self, keypoints, score, dt : Profile):
        
        kp = toBoneboxCoord(keypoints, norm=True) - 0.5    # normalization and centralization
        out, time = self.walkAe.predictSingleCap(kp, score, None, normed=True)
        conf = [out[:3].sum(), out[3]]
        out, time = self.phoneAe.predictSingleCap(kp, score, None, normed=True) 
        dt.t += time
        conf = [*(conf[0] * out), conf[1]]    # conf: (walking, play with one, play with two, other)
        
        return conf
    
    def predictMultiCaps(self, tvc, dt : Profile):
        '''
            phoneWalking Action Astimation Of Multi Caps
            params:
                tvc:  points and score in shape `(t, v, c)` where
                    t : inputs sequence (time steps).,
                    v : number of graph node (body parts).,
                    c : channel (x, y, score).
                    
        '''
        tvc = np.array(tvc)

        for i,vc in enumerate(tvc):
            tvc[i][:,:2] = toBoneboxCoord(vc[:,:2], norm=True) - 0.5    # normalization and centralization
        out, time = self.walkAe.predict(np.array(tvc), None, normed=True)
        dt.t += time
        conf = [out[:3].sum(), out[3]]  # conf: (walk, not walk)  
        
        out, time = self.phoneAe.predict(tvc, None, normed=True)   
        dt.t += time
        conf = [*(conf[0] * out), conf[1]]    # conf: (walk, play with one, play with two, other)
        
        return conf

    # def predictSingleCap(self, keypoints, score, dt : Profile):
    #     kp = toBoneboxCoord(co2cocut(keypoints, [17, 2]), norm=True) # normalizied keypoints according to skeletion box
    #     walkingActionEstimation = self.walkAe
    #     out, time = walkingActionEstimation.predictSingleCap(kp, co2cocut(score, [17,1]), None, normed=True)
    #     dt.t += time
    #     conf = [out[[0, 1, 4]].sum(), out[[2, 3, 5, 6]].sum()]  # conf: (walk, not walk)  
    #     # sit = walkingActionEstimation.getLabel(out)
    #     # if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
    #     #     return False, conf
        
    #     kp = toBoneboxCoord(keypoints, norm=True) - 0.5    # normalization and centralization
    #     phoneActionEstimation = self.phoneAe
    #     out, time = phoneActionEstimation.predictSingleCap(kp, score, None, normed=True) 
    #     dt.t +=time
    #     conf = [*(conf[0] * out), conf[1]]    # conf: (walking, play with one, play with two, other)
    #     # phone = phoneActionEstimation.getLabel(phone)
    #     # if phone == 'nohand':
    #     #     return False
        
    #     return conf
    
    # def predictMultiCaps(self, tvc, dt : Profile):
    #     '''
    #         phoneWalking Action Astimation Of Multi Caps
    #         params:
    #             tvc:  points and score in shape `(t, v, c)` where
    #                 t : inputs sequence (time steps).,
    #                 v : number of graph node (body parts).,
    #                 c : channel (x, y, score).
                    
    #     '''
    #     tvc = np.array(tvc)

    #     cococutTvc = []
    #     for vc in tvc:
    #         vc = co2cocut(vc, [17, 3])
    #         vc[:,:2] = toBoneboxCoord(vc[:,:2], norm=True)
    #         cococutTvc.append(vc)
    #     out, time = self.walkAe.predict(np.array(cococutTvc), None, normed=True)
    #     dt.t += time
    #     conf = [out[:3].sum(), out[3]]  # conf: (walk, not walk)  
    #     # sit = walkingActionEstimation.getLabel(sit) 
    #     # if sit in ['Sitting', 'Lying Down', 'Sit down', 'Fall Down']:
    #     #     return False
        
    #     for i,vc in enumerate(tvc):
    #         tvc[i][:,:2] = toBoneboxCoord(vc[:,:2], norm=True) - 0.5    # normalization and centralization
    #     out, time = self.phoneAe.predict(tvc, None, normed=True)   
    #     dt.t += time
    #     conf = [*(conf[0] * out), conf[1]]    # conf: (walk, play with one, play with two, other)
    #     # phone = phoneActionEstimation.getLabel(phone)
    #     # if phone == 'nohand':
    #     #     return False
        
    #     return conf

