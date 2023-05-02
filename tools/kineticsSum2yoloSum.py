from pathlib import Path
import json
from tqdm import tqdm
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory: D:\_NewCode\PythonPro\Phone_Walking_Detector
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.PoseTransformer import readJson


def transform(
    name = 'Mscoco',
    parts = ['train', 'val'],
    input_root_dir = 'D:/_NewCode/PythonPro/st_gcn/st-gcn/stgcnTrainData/Mscoco/exp10',
    output_dir = 'datasets/yolodata/train_val',
):

    input_root_dir = Path(input_root_dir)
    output_dir = Path(output_dir)

    for part in parts:
        input_json = f'{name}_{part}_label.json'
        inputSumDict= readJson(input_root_dir / input_json)
        outputPath = output_dir / f'{part}.txt'
        with outputPath.open('w') as f:
            for filename in tqdm(inputSumDict, desc=part):
                classStr = inputSumDict[filename]['label']
                f.write(f'./images/{classStr}/{filename}.jpg\n')


if __name__ == '__main__':
    transform()
