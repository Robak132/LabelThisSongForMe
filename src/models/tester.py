import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.common import move_to_cuda, get_auc, load_model, Statistics, Config, current_time, \
    convert_mp3_to_npy, get_data_chunked


class Tester:
    def __init__(self, config: Config = None, cuda: bool = None, mode: str = "TEST"):
        if config is None:
            config = Config()
        self.sr = config.sr

        # model path and step size
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step

        # cuda
        self.is_cuda = torch.cuda.is_available() if cuda is None else cuda
        print(f"[{current_time()}] Tester initialised with CUDA: {self.is_cuda} and mode: {mode}")

        # model
        self.model = move_to_cuda(config.model)
        self.loss_function = nn.BCELoss()

        self.tags = np.load(config.tags_path, allow_pickle=True)
        self.binary = np.load(config.binary_path, allow_pickle=True)
        if mode == "VALID":
            self.test_list = np.load(config.valid_path, allow_pickle=True)
        else:
            self.test_list = np.load(config.test_path, allow_pickle=True)
        self.data_path = config.data_path
        self.input_length = config.input_length

    def test(self, model=None):
        if model is None:
            model = load_model(self.model_save_path, self.model)

        est_array = []
        gt_array = []
        losses = []
        results = []
        model.eval()
        for ix, mp3_path in tqdm(self.test_list):
            npy_path = os.path.join(self.data_path, 'mtat', 'npy', mp3_path.split('/')[0],
                                    mp3_path.split('/')[1][:-3]) + 'npy'
            npy_data = np.load(npy_path, mmap_mode='c')

            ground_truth = self.binary[int(ix)]
            y = torch.tensor([ground_truth.astype('float32') for _ in range(len(npy_data) // self.input_length)])
            out = self.predict_npy(npy_data, model)
            loss = self.loss_function(out, y)

            losses.append(float(loss))
            results.append(out)
            est_array.append(np.array(out).mean(axis=0))
            gt_array.append(ground_truth)
        mean_loss = np.mean(losses)
        roc_auc, pr_auc = get_auc(np.array(est_array), np.array(gt_array))
        print(f"[{current_time()}] Loss/valid: {mean_loss:.4f}")
        print(f"[{current_time()}] AUC/ROC: {roc_auc:.4f}")
        print(f"[{current_time()}] AUC/PR: {pr_auc:.4f}")
        return results, Statistics(roc_auc, pr_auc, mean_loss)

    def predict_tags(self, mp3_file=None, model=None) -> pd.DataFrame:
        if model is None:
            model = load_model(self.model_save_path, self.model)
        if mp3_file is not None:
            out = self.predict_mp3(mp3_file, model).T
            df = pd.DataFrame(out, index=self.tags)
            df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
            return df

    def predict_mp3(self, x, model=None):
        if model is None:
            model = load_model(self.model_save_path, self.model)

        return self.predict_npy(convert_mp3_to_npy(x, self.sr), model)

    def predict_npy(self, x, model=None):
        if model is None:
            model = load_model(self.model_save_path, self.model)

        x = get_data_chunked(x, self.input_length)
        x = move_to_cuda(x)
        out = model(x).detach().cpu()
        return out
