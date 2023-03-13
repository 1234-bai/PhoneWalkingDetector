import argparse
import numpy as np
import cv2
from pathlib import Path

from libs.yolov5.yolov5DetectorApi import TargetsDecetor, TargetsAnnotator
from libs.yolov5.utils.plots import colors, save_one_box
from libs.yolov5.utils.general import check_requirements, increment_path, print_args
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer
from libs.st_gcn.StgcnApi import ActionEstimation

from MathUtils import twoPointsSuperpose, getBoxCenters


def run(
    source = 'images',
    device = 0,
    view_img = True,
    line_thickness = 2,
    nosave = False,
    save_crop = True,
    save_dir = 'runs/test',
    name = 'exp',
    exist_ok = False,
    vid_stride = 1

):

    ae = ActionEstimation(weight_file='libs\st_gcn\model\st-gcn-tsstg-fail-model.pth')

    poseTest = SingleImagePoseEstimation(
        configFilePath='libs\\Alphapose\\configs\\coco\\resnet\\256x192_res50_lr1e-3_1x.yaml',
        checkpoint='libs\\Alphapose\\pretrained_models\\fast_res50_256x192.pth',
        device=device
    )

    test =  TargetsDecetor(
        weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\yolov5s.pt',
        data='libs\\yolov5\\data\\coco128.yaml',
        device=device
    )
    dataset = test.loadData(source=source,vid_stride=vid_stride)

    phoneTest= TargetsDecetor(
        weights='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\weights\phone_ep20.pt',
        data='D:\_NewCode\PythonPro\Phone_Walking_Detector\libs\yolov5\data\phone.yaml',
        device=device
    )

    save = not nosave
    saveDir = increment_path(save_dir+'/'+name, exist_ok=exist_ok, mkdir=True) if save or save_crop else None
    preFilename = ''
    videoWriter : cv2.VideoWriter = None
    for path, _, im0s, vid_cap, _ in dataset:

        # 获得文件名字和后缀
        if dataset.mode == 'stream':
            filename = path[0]
            suffix = 'mp4'
        else:
            filename = Path(path).stem
            suffix = Path(path).suffix

        # 获得原始图片
        im0 = im0s[0] if dataset.mode == 'stream' else im0s
        img = im0.copy()

        # 注释器（画图器）
        annotator = TargetsAnnotator(img, line_thickness)

        # 检测人像
        _, peopleXyxyBoxes, crops, confs = test.detectorSingleImg(img, classes=[0])
        if(len(peopleXyxyBoxes) > 0):

            # 根据人像检测骨骼结点
            poses = poseTest.process(path, img, peopleXyxyBoxes, confs) # 获得骨骼结点 list of 'keypoints:list , scores:list, box: list of 4}' index is people_number

            # 对每个人像进行手机检测和动作检测
            handsIndex = poseTest.getHandIndex() # 获得该骨骼结点格式的手部结点编号
            earsIndex = poseTest.getEarIndex()  # 获得该骨骼结点格式的耳朵结点编号
            for i,crop in enumerate(crops): #   对于每个人像
                keypoints = poses[i]['keypoints'] # 获得此人的骨骼结点
                handCenters = [keypoints[handsIndex[0]], keypoints[handsIndex[1]]] # 获得手部结点
                earCenters = [keypoints[earsIndex[0]], keypoints[earsIndex[1]]]
                scores = poses[i]['kp_score'] # 获得此人的骨骼结点置信度
                
                # 动作检测
                # 骨骼结点格式转换，与动作检测模型的骨骼结点输入格式匹配
                keypoints = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(keypoints, [17, 2])
                scores = AlphaposeDataTransformer.coco2017Keypoints2CocoCut(scores, [17, 1])
                box = poses[i]['bbox'] # 每个人像的xywhBox
                actionName = ae.predict(keypoints, scores, (box[2], box[3]))
                if(actionName != 'Walking'):
                    continue

                # 手机检测
                _, phoneXyxyBoxes,_, _ = phoneTest.detectorSingleImg(crop, classes=[0], conf_thres=0.4)
                if(len(phoneXyxyBoxes)):   # 人像图中存在手机
                    phoneCenters = getBoxCenters(phoneXyxyBoxes)
                    for j, pc in enumerate(phoneCenters):  # 对于每个手机，是否与人手重合
                        for hc in handCenters:
                            if twoPointsSuperpose(pc, hc, crop.shape): # 重合则开始验证动作
                                peopleBox = peopleXyxyBoxes[i]
                                if save_crop:
                                    cropPath = increment_path(saveDir / 'crop'/ (filename+'.jpg'),sep='_')
                                    save_one_box(peopleBox, im0, file=cropPath, BGR=True)
                                annotator.box_label(peopleBox, label=actionName, color=colors(0))   # 深拷贝，会直接在原始图片上进行修改
                                annotator.double_box_label(peopleBox, phoneXyxyBoxes[j], label='phone', color=colors(5))
                                break

        img = annotator.result()
        filename = filename + '.' + suffix
        # save image/video
        if save:
            savePath = saveDir.joinpath(filename)
            if dataset.mode == 'image':
                cv2.imwrite(savePath, img)
            else:   # stream or vedio
                if filename == preFilename: # 是前一个视频的下一帧
                    assert(videoWriter != None)
                    videoWriter.write(img)
                else:   # 是一个新的视频或者第一个视频
                    preFilename = filename
                    if videoWriter is not None:
                        videoWriter.release() # release previous video writer
                        videoWriter = None
                    if vid_cap: #video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:   # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    videoWriter = cv2.VideoWriter(str(savePath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        # view image
        if view_img:  
            cv2.imshow(filename, img)  
            if cv2.waitKey(-1) & 0xFF == ord('q'):
                break
    if videoWriter is not None:
        videoWriter.release()
        videoWriter = None

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--nosave', action='store_true', help='save images/videos result')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-dir', default='runs/test', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)