import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np


def bar_fun(x, y, ax, color, label):
    ax.bar(x, y, width=0.01, color=color, label=label)
    ax.legend()


def line_fun(x, y, ax, color, label):
    s = y.sum()
    data = [1.0 * y[:i].sum() / s for i in range(len(x))]
    ax.plot(x, data, color = color, linewidth=4, label=label)
    ax.legend()


def drawWristAndEar_Single(wristExs, earExs, ax, ylabel):
    wristExs.sort()
    earExs.sort()
    x, y = np.unique(wristExs, return_counts=True)
    x = x[y > 3][1:]
    y = y[y > 3][1:]
    line_ax = [i.twinx() for i in ax]
    bar_fun(x, y, ax[0], 'blue', 'wrist')
    line_fun(x, y, line_ax[0], 'red', 'wrist')
    x,y = np.unique(earExs, return_counts=True)
    x = x[y > 5][1:]
    y = y[y > 5][1:]
    bar_fun(x, y, ax[1], 'blue', 'ear')
    line_fun(x, y, line_ax[1], 'red', 'ear')
    ax[0].set_ylabel(ylabel)

def drawWristAndEar_Two(wristExs, earExs, bar_ax, line_ax, ylabel):
    wristExs.sort()
    earExs.sort()
    x, y = np.unique(wristExs, return_counts=True)
    x = x[y > 3][1:]
    y = y[y > 3][1:]
    bar_fun(x, y, bar_ax[0], 'red', 'wrist')
    line_fun(x, y, line_ax[0], 'red', 'wrist')
    x,y = np.unique(earExs, return_counts=True)
    x = x[y > 5][1:]
    y = y[y > 5][1:]
    bar_fun(x, y, bar_ax[1], 'blue', 'ear')
    line_fun(x, y, line_ax[1], 'blue', 'ear')
    bar_ax[0].set_ylabel(ylabel)
    line_ax[0].set_ylabel(ylabel)

def run(
    source_dir,
    class_files,
    save_dir,
):
    source_dir = Path(source_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / 'Phone.png')
    count = len(class_files)
    fig, ax = plt.subplots(count+1, 2, sharex=False, sharey=False, figsize=(18, 16))
    total_wrists = []
    total_ears = []
    for i, cl in tqdm(enumerate(class_files)):
        source = source_dir / (cl+'.txt')
        with source.open("r") as f:
            data = f.readlines()
            wristExs = np.array(list(map(float, data[1].split(' '))))
            earExs = np.array(list(map(float, data[3].split(' '))))
        total_wrists.append(wristExs)
        total_ears.append(earExs)
        drawWristAndEar_Single(wristExs, earExs, ax[i], cl)
    total_wrists = np.concatenate(total_wrists, axis=0)
    total_ears = np.concatenate(total_ears, axis=0)
    drawWristAndEar_Single(total_wrists, total_ears, ax[count], 'total')
    fig.savefig(save_path)
    plt.close(fig)




if __name__ == '__main__':
    run(
        source_dir='runs/phone',
        class_files=['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'Stand', 'Sit', 'Other'],
        save_dir='runs/phone'
    )