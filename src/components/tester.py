import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from components.predictor import Predictor
from src.components.common import move_to_cuda, get_auc, Statistics, Config, current_time


class BaseTester:
    def __init__(self, config: Config = None, model_filename: str = None, cuda: bool = None):
        if config is None:
            config = Config()

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.get_name())
        self.model_filename = model_filename



class Tester:
    def __init__(self, config: Config = None, model_filename: str = None, cuda: bool = None, mode: str = "TEST"):
        if config is None:
            config = Config()

        valid_path = os.path.join(config.dataset_split_path, config.dataset_name, "valid.npy")
        test_path = os.path.join(config.dataset_split_path, config.dataset_name, "test.npy")
        binary_path = os.path.join(config.dataset_split_path, config.dataset_name, "binary.npy")

        self.predictor = Predictor(config, model_filename, cuda)

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.get_name())
        self.model_filename = model_filename

        # cuda
        self.is_cuda = torch.cuda.is_available() if cuda is None else cuda
        print(f"[{current_time()}] Tester initialised with CUDA: {self.is_cuda} and mode: {mode}")

        # model
        self.model = move_to_cuda(config.model)
        self.log_step = config.log_step
        self.loss_function = nn.BCELoss()

        self.binary = {row[0]: row[1:] for row in np.load(binary_path, allow_pickle=True)}
        if mode == "VALID":
            self.test_list = np.load(valid_path, allow_pickle=True)
        else:
            self.test_list = np.load(test_path, allow_pickle=True)
        self.data_path = config.data_path
        self.input_length = config.input_length

    def test(self, model) -> Statistics:
        est_array = []
        gt_array = []
        losses = []
        model.eval()
        for ix, mp3_path in tqdm(self.test_list):
            npy_path = os.path.join(self.data_path, 'mtat/npy', mp3_path.split('/')[0], mp3_path.split('/')[1][:-3]) + 'npy'
            npy_data = np.load(npy_path, mmap_mode='c')

            ground_truth = self.binary[int(ix)]
            y = np.tile(ground_truth, (len(npy_data) // self.input_length, 1))
            y = torch.tensor(y.astype("float32"))
            out = self.predictor.predict_data(npy_data, model)
            loss = self.loss_function(out, y)

            losses.append(float(loss))
            est_array.append(np.array(out).mean(axis=0))
            gt_array.append(ground_truth)
        mean_loss = np.mean(losses)
        roc_auc, pr_auc = get_auc(np.array(est_array), np.array(gt_array))
        print(f"[{current_time()}] Loss/valid: {mean_loss:.4f}")
        print(f"[{current_time()}] AUC/ROC: {roc_auc:.4f}")
        print(f"[{current_time()}] AUC/PR: {pr_auc:.4f}")
        return Statistics(roc_auc, pr_auc, mean_loss)
