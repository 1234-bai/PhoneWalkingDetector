from pathlib import Path
from tqdm import tqdm

input_dir = Path('datasets\phoneData\project-3-at-2023-04-20-22-32-ba0bab70\labels')
label_map = [0, 0, 0, 1, 2]
class_map = {
    str(i) : str(d) for i, d in enumerate(label_map)
}


for file in tqdm(Path.iterdir(input_dir)):
    ss = []
    with file.open("r") as f:
        cls_strs = f.readlines()
        for cls_str in cls_strs:
            ss.append(list(cls_str))
    for i, s in enumerate(ss):
        ss[i][0] = class_map[s[0]]
    with file.open("w") as f:
        for s in ss:
            f.write(''.join(s))
