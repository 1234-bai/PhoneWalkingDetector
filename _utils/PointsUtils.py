import numpy as np

def pointInBox(p, box, ex=0.5):
    p = [int(x) for x in p]
    box = np.array(box)
    c = ((box[2:] - box[:2]) * ex)[[0, 1, 0, 1]] * np.array([-1, -1, 1, 1])
    box = [int(x) for x in (box+c)]
    return ((p[0] - box[0]) * (p[0] - box[2])) <= 0 and ((p[1] - box[1]) * (p[1] - box[3])) <= 0


def pointsAnyInBox(points, box, ex=0.5):
    '''
        if any point of points in the box return True
    '''
    for point in points:
        if pointInBox(point, box, ex): return True
    return False

def getBoxCenters(boxes):
    boxes = np.array(boxes)
    return np.concatenate(
        (
            np.array([(boxes[:, 2] + boxes[:, 0])/2.0]).T,
            np.array([(boxes[:, 3] + boxes[:, 1])/2.0]).T
        ),
        axis=1
    )

def kepoints2bbox(kpt, ex=5):
    """Get bbox that hold on all of the points (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def toBoneboxCoord(keypoints, norm=False, extension=5):
    '''
        trans keypoints to their boneBox coord
    '''
    boneBox = kepoints2bbox(keypoints, extension) # 求出骨架盒子
    # wh =  boneBox[2:]- boneBox[:2] # 求出骨架盒子宽和高
    # 求出相对于骨架盒子的(归一化)坐标
    kt = (keypoints - boneBox[:2]) / (boneBox[2:]- boneBox[:2]) if norm else (keypoints - boneBox[:2])
    return kt
