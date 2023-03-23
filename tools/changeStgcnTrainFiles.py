import argparse
from pathlib import Path
from _utils.PoseTransformer import readJson, writeJson, coco2017Kps2coco2017cut
import json
from tqdm import tqdm

def transform(
    new_class = ['phone', 'standing'],
    oldClass2newClassMap = [0, 0, 0, 0, 1, -1, 1],
    oldClassCountThres = ([-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]),  # 旧有类别限定数量
    skeleonTransformer = None,
    project_name = 'Mscoco',
    parts = ['train', 'val'],
    input_root_dir = 'D:/_NewCode/PythonPro/st_gcn/st-gcn/stgcnTrainData/Mscoco/exp6',
    output_dir = f'stgcnTrainData/',
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
            filename = file.stem
            d = readJson(file)
            label_index = d['label_index']
            newLabelIndex = oldClass2newClassMap[label_index]
            if newLabelIndex == -1:
                if filename in inputSumDict:
                    del inputSumDict[filename]
                continue
            if classCountThres[label_index] != 0:
                classCountThres[label_index] -= 1
                newLabel = classNames[newLabelIndex]
                d['label_index'] = newLabelIndex
                d['label'] = newLabel
                if skeleonTransformer:
                    for kpDict in d['data']:
                        sk = kpDict['skeleton']
                        for psAdsc  in sk:
                            psAdsc['pose'] = skeleonTransformer[0](psAdsc['pose'])
                            psAdsc['score'] = skeleonTransformer[1](psAdsc['score'])
                inputSumDict[filename]['label_index'] = newLabelIndex
                inputSumDict[filename]['label'] = newLabel
                writeJson(d, output_dir / input_dir_name, filename + '.json')
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
        oldClassCountThres=([-1, -1, 380, -1, -1, -1, -1], [-1, -1, 90, -1, -1, -1, -1]),
        input_root_dir = 'D:/_NewCode/PythonPro/st_gcn/st-gcn/stgcnTrainData/Mscoco/exp10',
        # skeleonTransformer = [kineticsFormatMscocoKp2MscococutKp, coco2017Kps2coco2017cut]
    )
