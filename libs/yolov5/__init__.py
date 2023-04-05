import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.torch_utils import select_device
from utils.plots import colors, save_one_box
from utils.general import check_file, check_requirements, increment_path, print_args, LOGGER, Profile
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams

def loadData(
    source,  # file/dir/URL/glob/screen/0(webcam)
    vid_stride=1  # video frame-rate stride 帧率：视频每多少帧截取一次):
):
    # 区分数据来源 
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
        
    # 加载数据
    if webcam:
        dataset = LoadStreams(source, vid_stride=vid_stride)
    elif screenshot:
        dataset = LoadScreenshots(source)
    else:
        dataset = LoadImages(source, vid_stride=vid_stride)

    return dataset


__all__ = [
    select_device, loadData, 
    colors, save_one_box, increment_path, 
    LOGGER, Profile, print_args, check_requirements,
]