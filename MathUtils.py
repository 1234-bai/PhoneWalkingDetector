import numpy as np

def twoPointsSuperpose(p1, p2, imageSize, superposeThresold = 0.4):
    # if(isinstance(p1, list)):
    #     assert(len(p1) == 2 and len(p2) == 2)
    # width = 1.0 * (p2[0] - p1[0]) / imageSize[0]
    # height = 1.0 * (p2[1] - p1[1]) / imageSize[1]
    # return width * width + height * height <= superposeThresold * superposeThresold
    return True


def getBoxCenters(boxes):
    boxes = np.array(boxes)
    return np.concatenate(
        (
            np.array([(boxes[:, 2] + boxes[:, 0])/2.0]).T,
            np.array([(boxes[:, 3] + boxes[:, 1])/2.0]).T
        ),
        axis=1
    )