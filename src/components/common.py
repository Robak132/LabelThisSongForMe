from dataclasses import dataclass
from datetime import datetime
from typing import Union

import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from numpy import ndarray
from plotly.graph_objs import Bar, Figure, Layout
from sklearn import metrics
from sklearn.base import ClassifierMixin
from torch import tensor, Tensor
from torch.nn import Module

from components.preprocessor import BasePreProcessor
from src.external.musicnn import Musicnn


@dataclass
class Statistics:
    def __init__(self, roc_auc, pr_auc, mean_loss, f1_score):
        self.roc_auc = roc_auc
        self.pr_auc = pr_auc
        self.mean_loss = mean_loss
        self.f1_score = f1_score


@dataclass
class Config:
    preprocessor: BasePreProcessor = None
    model: Union[Module, ClassifierMixin] = Musicnn()
    n_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    model_filename_path: str = "models"
    data_path: str = 'data'
    log_step: int = 100
    sr: int = 16000
    input_length: int = 3 * sr
    dataset_split_path: str = "split"
    dataset_name: str = "mtat"
    logs_path: str = "logs"


def get_metrics(est_array, gt_array):
    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    f1_score = metrics.f1_score(gt_array, est_array >= 0.5, average='macro')
    return roc_aucs, pr_aucs, f1_score


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
    x, _ = librosa.load(mp3_file, sr=sr)
    return x


def create_tagogram(prediction: pd.DataFrame):
    fig = px.imshow(img=prediction.values,
                    y=prediction.index,
                    color_continuous_scale='RdBu_r',
                    title="Tag probability in each chunk")
    return fig


def plot_probability_graph(prediction: pd.DataFrame):
    fig = Figure(data=Bar(x=prediction.mean(axis=1), y=prediction.index, orientation='h'),
                 layout=Layout(title="Mean tag probability"))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def load_file_lists(file_lists: list[str]) -> ndarray:
    files = []
    for file_list in file_lists:
        for obj_file in np.load(file_list, allow_pickle=True):
            files.append(obj_file)
    return np.array(files)
