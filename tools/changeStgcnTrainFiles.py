from pathlib import Path
import json
from tqdm import tqdm
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from _utils.PoseTransformer import readJson, writeJson, coco2017Kps2coco2017cut


def transform(
    new_class = ['phone', 'standing'],
    oldClass2newClassMap = [0, 0, 0, 0, 1, -1, 1],
    oldClassCountThres = ([-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]),  # 旧有类别限定数量
    skeleonTransformer = None,
    project_name = 'Mscoco',
    parts = ['train', 'val'],
    input_root_dir = 'D:/_NewCode/PythonPro/st_gcn/st-gcn/stgcnTrainData/Mscoco/exp6',
    output_dir = f'datasets/stgcnTrainData/',
):
    
    classNames = new_class
    name = project_name
    input_root_dir = Path(input_root_dir)
    output_dir = Path(output_dir) / name / 'out'

    for i, part in enumerate(parts):
        input_dir_name = f'{name}_{part}'
        input_dir = input_root_dir /  input_dir_name
        input_json = f'{name}_{part}_label.json'
        inputSumDict= readJson(input_root_dir / input_json)
        classCountThres = oldClassCountThres[i]
        for file in tqdm(Path.iterdir(input_dir), desc=part):
            filename = file.stem # 获得无后缀名的文件名称
            d = readJson(file)
            label_index = d['label_index']  # 获得旧类别号
            newLabelIndex = oldClass2newClassMap[label_index] # 获得新类别号
            if newLabelIndex == -1: # 新类别号==-1，说明被删除
                del inputSumDict[filename]  # 在总结json文件中删除
                continue
            if classCountThres[label_index] != 0: # 新类别的限制数量为正值
                classCountThres[label_index] -= 1
                newLabel = classNames[newLabelIndex]    #获得新类别名称
                d['label_index'] = newLabelIndex    # 修改类别号为新类别号
                d['label'] = newLabel   # 修改名称为新类别名称
                if skeleonTransformer:  # 如果有骨骼结点转换函数
                    for kpDict in d['data']:
                        sk = kpDict['skeleton']
                        for psAdsc  in sk:
                            psAdsc['pose'] = skeleonTransformer[0](psAdsc['pose'])
                            psAdsc['score'] = skeleonTransformer[1](psAdsc['score'])
                inputSumDict[filename]['label_index'] = newLabelIndex
                inputSumDict[filename]['label'] = newLabel
                writeJson(d, output_dir / input_dir_name, filename + '.json')
            else:
                del inputSumDict[filename]  # 在总结json文件中删除
        writeJson(inputSumDict, output_dir, input_json)


def recoverJsonDirFromJsonfile(
    name = 'Mscoco',
    parts = ['train', 'val'],
    input_root_dir = 'D:/_NewCode/PythonPro/st_gcn/st-gcn/stgcnTrainData/Mscoco',
):
    for part in parts:
        input_dir_name = f'{name}_{part}'
        input_dir = input_root_dir /  input_dir_name
        input_json = f'{name}_{part}_label.json'
        inputSumDict= readJson(input_root_dir / input_json)

        for file in Path.iterdir(input_dir):
            filename = file.stem
            d = readJson(file)
            d['label_index'] = inputSumDict[filename]['label_index']
            d['label'] = inputSumDict[filename]['label']
            with file.open('w') as f:
                json.dump(d, f)


def kineticsFormatMscocoKp2MscococutKp(keypoints):
    return keypoints[:13*2]


if __name__ == '__main__':
    transform(
        new_class=['call', 'oneHand', 'twoHands', 'stand'],
        oldClass2newClassMap=[0, 1, 2, -1, 3, -1, -1],
        oldClassCountThres=([-1, 360, 360, -1, -1, -1, -1], [-1, -1, 90, -1, 90, -1, -1]),
        input_root_dir = 'D:/_NewCode/PythonPro/st_gcn/st-gcn/stgcnTrainData/Mscoco/exp6',
        # skeleonTransformer = [kineticsFormatMscocoKp2MscococutKp, coco2017Kps2coco2017cut]
    )
