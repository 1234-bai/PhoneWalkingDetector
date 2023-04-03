import cv2
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from libs.yolov5.yolov5DetectorApi import TargetsDetector, select_device
from libs.Alphapose.AlphaposeApi import SingleImagePoseEstimation, AlphaposeDataTransformer as ADT
from _utils.PoseTransformer import writeJson, readJson, alphaose2kineticsFormat

def jsonPosePack(poses, label_index, label_name, copyTimes):
    data = []
    if poses is not None and len(poses) > 0:
        skeleton = alphaose2kineticsFormat(poses, True)
        data = [{
            "frame_index" : i,
            "skeleton" : skeleton
        } for i in range(copyTimes)]
    return {
        "data" : data,
        "label" : label_name,
        "label_index" : label_index 
    }

def writePoseJson(poses, label_index, label_name, copyTimes, jsonPath : Path, filename):
    poseDict = jsonPosePack(poses, label_index, label_name, copyTimes)
    writeJson(poseDict, jsonPath, filename)


name = 'Mscoco'
input_dir = Path("D:/QianXiaoYi/Pictures/Data/normal_images")
output_dir = Path(f'stgcnTrainData/{name}')
output_train_json_dir = output_dir / (f'{name}_train')
output_val_json_dir = output_dir / (f'{name}_val')
output_train_json = f'{name}_train_label.json'
output_val_json = f'{name}_val_label.json'

valThres = 8
outTrainSumJson = readJson(output_dir / output_train_json)
outValSumJson = readJson(output_dir / output_val_json)
label_names = ['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'Photograph', 'Stand', 'Sit', 'Other'] # 根据动作分类，而不是手机出现的位置
copy_dir = [(output_dir / x) for x in label_names]
for x in copy_dir:
    x.mkdir(parents=True, exist_ok=True)
frameCount = 30

device = select_device(0)
peopleDec =  TargetsDetector(
    weights='D:/_NewCode/PythonPro/Phone_Walking_Detector/libs/yolov5/weights/yolov5s.pt',
    data='libs/yolov5/data/coco128.yaml',
    device=device
)

# poseEst = SingleImagePoseEstimation(
#     configFilePath='libs/Alphapose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
#     checkpoint='libs/Alphapose/pretrained_models/halpe26_fast_res50_256x192.pth',
#     device=0
# )

poseEst = SingleImagePoseEstimation(
    configFilePath='libs/Alphapose/configs/coco_256x192_res50_lr1e-3_1x.yaml',
    checkpoint='libs/Alphapose/pretrained_models/fast_res50_256x192.pth',
    device=device
)

dataset = peopleDec.loadData(source=input_dir)
count = 0
for path, _, im0s, vid_cap, s in dataset:

    im0 = im0s
    _, peoXyxyBoxes, _, confs, _= peopleDec.detectorSingleImg(im0, classes=[0], conf_thres=0.45)
    file = Path(path)
    if(len(peoXyxyBoxes) > 0) :
        poses,_ = poseEst.process(im0, peoXyxyBoxes, confs)
        filename = file.stem
        im = ADT.viewpPoseInImage(im0, poses, poseEst.getVisThres())
        cv2.imshow(filename, im)
        choice =  cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if choice == ord('b'): break
        if choice == ord('d'):
            if filename in outValSumJson:
                del outValSumJson[filename]
                (output_val_json_dir / (filename+'.json')).unlink()
            if filename in outTrainSumJson:
                del outTrainSumJson[filename]
                (output_train_json_dir / (filename+'.json')).unlink()             
            file.unlink()
            continue
        for i, label in enumerate(label_names):
            if choice == ord(str(i)):
                count += 1
                imgDict = {
                    "has_skeleton": poses is not None and len(poses) > 0, 
                    "label": label, 
                    "label_index": i
                }
                if(count % 10 < valThres):    # train set
                    print(f'train:{count}')
                    writePoseJson(poses, i, label, frameCount, output_train_json_dir, (filename+'.json'))
                    outTrainSumJson[filename] = imgDict
                    if filename in outValSumJson:
                        writePoseJson(poses, i, label, frameCount, output_val_json_dir, (filename+'.json'))
                        outValSumJson[filename] = imgDict
                        print(f'{filename} also in val')
                else:   #   val set, last 400 images
                    print(f'val:{count}')
                    writePoseJson(poses, i, label, frameCount, output_val_json_dir, (filename+'.json'))
                    outValSumJson[filename] = imgDict
                    if filename in outTrainSumJson:
                        writePoseJson(poses, i, label, frameCount, output_train_json_dir, (filename+'.json'))
                        outTrainSumJson[filename] = imgDict
                        print(f'{filename} also in train')
                assert(cv2.imwrite(copy_dir[i] / file.name, im0))
                # file.unlink()
                break

writeJson(outTrainSumJson, output_dir, output_train_json)
writeJson(outValSumJson, output_dir, output_val_json)
