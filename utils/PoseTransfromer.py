import torch

class PoseDataTransformer():

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
    
    @staticmethod
    def alphaose2kineticsFormat(poses):
        pass