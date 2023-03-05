import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt

from models.common import Config
from models.tester import Tester

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to tag music using selected model')
    parser.add_argument('filename', type=str, help='Path to music file (MP3)')
    parser.add_argument('model', type=str, help='Path to model file (PTH)')
    args = parser.parse_args()

    tester = Tester(Config(batch_size=None))
    out = tester.predict_mp3(args.filename)
    print(f"File: {args.filename}")

    mean_out = torch.mean(out, dim=0)
    tags = [[tester.tags[i], mean_out[i].item()] for i in range(len(mean_out))]
    tags.sort(key=lambda x: x[1], reverse=True)
    tags = np.array(tags)
    print(f"Tags: {tags}")

    fig, ax = plt.subplots()
    img = ax.imshow(out.T, aspect='auto')
    ax.set_yticks(np.arange(len(tester.tags)), labels=tester.tags)
    ax.xaxis.set_visible(False)
    plt.colorbar(img, ax=ax)

    fig, ax = plt.subplots()
    tags = tags.T
    ax.barh(tags[0], tags[1], align='center')
    ax.yaxis.set_ticks(np.arange(len(tags[0])), labels=tags[0])

    plt.show()
