import os
from array import array
from typing import Tuple, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import tensor
from tqdm import tqdm

from models.common import move_to_cuda, get_auc, load_model, Statistics, Config, current_time, \
    convert_mp3_to_npy, get_random_data_chunk, get_data_chunked
from models.loader import get_audio_loader


class Tester:
    def __init__(self, config: Config = None, mode: str = "TEST"):
        if config is None:
            config = Config()

        # data loader
        self.data_loader = get_audio_loader(data_path=config.data_path,
                                            batch_size=config.batch_size,
                                            files_path=config.valid_path if mode == "VALID" else config.test_path,
                                            binary_path=config.binary_path,
                                            num_workers=config.num_workers,
                                            input_length=config.input_length,
                                            shuffle=False)

        # model path and step size
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step

        # cuda
        self.is_cuda = torch.cuda.is_available()
        print(f"[{current_time()}] Tester initialised with CUDA: {self.is_cuda} and mode: {mode}")

        # model
        self.model = move_to_cuda(config.model)
        self.loss_function = nn.BCELoss()

        self.tags = np.load("split/mtat/tags.npy", allow_pickle=True)
        self.binary = np.load(config.binary_path, allow_pickle=True)
        self.test_list = np.load(config.test_path, allow_pickle=True)
        self.data_path = config.data_path
        self.batch_size = config.batch_size
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
            y = torch.tensor([ground_truth.astype('float32') for _ in range(self.batch_size)])
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

    def predict_tags(self, mp3_file=None, model=None) -> tuple[ndarray, ndarray, ndarray]:
        if model is None:
            model = load_model(self.model_save_path, self.model)
        if mp3_file is not None:
            out = self.predict_mp3(mp3_file, model)
            mean_out = torch.mean(out, dim=0)
            tags = [[self.tags[i], mean_out[i].item()] for i in range(len(mean_out))]
            tags.sort(key=lambda x: x[1], reverse=True)
            return out, self.tags, np.array(tags)

    def predict_mp3(self, x, model=None):
        if model is None:
            model = load_model(self.model_save_path, self.model)

        return self.predict_npy(convert_mp3_to_npy(x, 16000), model)

    def predict_npy(self, x, model=None):
        if model is None:
            model = load_model(self.model_save_path, self.model)

        x = get_data_chunked(x, self.input_length)
        x = move_to_cuda(x)
        out = model(x).detach().cpu()
        return out
