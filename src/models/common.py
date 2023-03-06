from dataclasses import dataclass
from datetime import datetime

import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn import metrics
from torch import tensor, Tensor
from torch.nn import Module

from external.model import Musicnn


@dataclass
class Statistics:
    def __init__(self, roc_auc, pr_auc, mean_loss):
        self.roc_auc = roc_auc
        self.pr_auc = pr_auc
        self.mean_loss = mean_loss


@dataclass
class Config:
    num_workers: int = 0
    model: Module = Musicnn()
    n_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-4
    model_save_path: str = "models/musicnn.pth"
    data_path: str = 'data'
    log_step: int = 100
    sr: int = 16000
    input_length: int = 3 * sr
    train_path: str = "split/mtat/train.npy"
    valid_path: str = "split/mtat/valid.npy"
    test_path: str = "split/mtat/test.npy"
    binary_path: str = "split/mtat/binary.npy"


def get_auc(est_array, gt_array):
    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    return roc_aucs, pr_aucs


def get_random_data_chunk(data, input_length) -> Tensor:
    random_idx = int(np.floor(np.random.random(1) * (len(data) - input_length)))
    return tensor(np.array(data[random_idx:random_idx + input_length]))


def get_data_chunked(data, input_length) -> Tensor:
    batch_size = len(data) // input_length
    return tensor(np.array([data[input_length * i: input_length * (i + 1)] for i in range(batch_size)]))


def move_to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def current_time() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def load_model(filename, model):
    s = torch.load(filename)
    if 'spec.mel_scale.fb' in s.keys():
        model.spec.mel_scale.fb = s['spec.mel_scale.fb']
    model.load_state_dict(s)
    model = move_to_cuda(model)
    return model


def convert_mp3_to_npy(mp3_file, sr) -> ndarray:
    x, sr = librosa.load(mp3_file, sr=sr)
    return x


def create_tagogram(raw_data, tags):
    fig, ax = plt.subplots()
    img = ax.imshow(raw_data.T, aspect='auto')
    ax.set_yticks(np.arange(len(tags)), labels=tags)
    ax.xaxis.set_visible(False)
    plt.colorbar(img, ax=ax)
    return fig


def plot_probability_graph(tags):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(tags))[::-1]
    ax.barh(y_pos, tags.T[1].astype('float64'), align='center')
    ax.set_yticks(y_pos, labels=tags.T[0])
    return fig
