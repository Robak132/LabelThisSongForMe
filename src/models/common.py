from dataclasses import dataclass
from datetime import datetime

import librosa
import numpy as np
import torch
from numpy import ndarray
from sklearn import metrics

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
    model: Musicnn = Musicnn(dataset='mtat')
    n_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-4
    model_save_path: str = "models/musicnn.pth"
    data_path: str = 'data'
    log_step: int = 100
    input_length: int = 3 * 16000
    train_path: str = "split/mtat/train.npy"
    valid_path: str = "split/mtat/valid.npy"
    test_path: str = "split/mtat/test.npy"
    binary_path: str = "split/mtat/binary.npy"


def get_auc(est_array, gt_array):
    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    return roc_aucs, pr_aucs


def get_tensor_chunked(data, input_length):
    random_idx = int(np.floor(np.random.random(1) * (len(data) - input_length)))
    x = np.array(data[random_idx:random_idx + input_length])
    return x


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
