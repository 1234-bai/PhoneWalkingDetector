import numpy as np

def twoPointsSuperpose(p1, p2, imageSize, superposeThresold = 0.4):
    # if(isinstance(p1, list)):
    #     assert(len(p1) == 2 and len(p2) == 2)
    # width = 1.0 * (p2[0] - p1[0]) / imageSize[0]
    # height = 1.0 * (p2[1] - p1[1]) / imageSize[1]
    # return width * width + height * height <= superposeThresold * superposeThresold
    return True


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

def pt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the points (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


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