from dataclasses import dataclass
from datetime import datetime

import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from numpy import ndarray
from plotly.graph_objs import Bar, Figure, Layout
from sklearn import metrics


@dataclass
class Statistics:
    def __init__(self, roc_auc, pr_auc, mean_loss, f1_score):
        self.roc_auc = roc_auc
        self.pr_auc = pr_auc
        self.mean_loss = mean_loss
        self.f1_score = f1_score


def get_metrics(est_array, gt_array):
    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    est_bin_array = np.where(est_array >= 0.5, 1, 0)
    f1_score = metrics.f1_score(gt_array, est_bin_array, average='macro')
    return roc_aucs, pr_aucs, f1_score


def move_to_cuda(x):
    try:
        if torch.cuda.is_available():
            x = x.cuda()
    except Exception:
        print("Cannot use cuda for model, defaulting to cpu")
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
    prediction = prediction.max(axis=1)
    prediction = prediction.sort_values()
    fig = Figure(data=Bar(x=prediction, y=prediction.index, orientation='h'),
                 layout=Layout(title="Mean tag probability"))
    return fig


def get_tags(prediction: pd.DataFrame):
    predicted_tags = prediction.max(axis=1)
    predicted_tags = predicted_tags.sort_values(ascending=True)
    predicted_tags = predicted_tags.apply(lambda x: 1 if x >= 0.5 else 0)
    predicted_tags = predicted_tags[predicted_tags == 1]
    predicted_tags = predicted_tags.index.to_list()
    return predicted_tags


def load_file_lists(file_lists: list[str]) -> ndarray:
    files = []
    for file_list in file_lists:
        for obj_file in np.load(file_list, allow_pickle=True):
            files.append(obj_file)
    return np.array(files)
