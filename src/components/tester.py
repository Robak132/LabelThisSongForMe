import os

import numpy as np
import torch
import torch.nn as nn
from torch import tensor
from tqdm import tqdm

from src.components.config import Config
from src.components.predictor import Predictor
from src.components.common import move_to_cuda, get_metrics, Statistics, current_time, load_model
from sklearn import metrics


class Tester:
    def __init__(self, config: Config = None, model_filename: str = None, cuda: bool = None, mode: str = "TEST"):
        if config is None:
            config = Config()

        valid_path = os.path.join(config.dataset_split_path, config.dataset_name, "valid.npy")
        test_path = os.path.join(config.dataset_split_path, config.dataset_name, "test.npy")
        binary_path = os.path.join(config.dataset_split_path, config.dataset_name, "binary.npy")

        self.predictor = Predictor(config, model_filename, cuda)

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.__class__.__name__, config.dataset_name)
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

    def get_metrics(self, est_array, gt_array):
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        f1_score = metrics.f1_score(gt_array, est_array >= 0.5, average='macro')
        return roc_aucs, pr_aucs, f1_score

    def test(self, model=None) -> Statistics:
        if model is None:
            model = load_model(os.path.join(self.model_filename_path, self.model_filename), self.model)

        est_array = []
        gt_array = []
        losses = []
        model.eval()
        for ix, mp3_path in tqdm(self.test_list):
            npy_path = os.path.join(self.data_path, 'mtat/npy', mp3_path.split('/')[0], mp3_path.split('/')[1][:-3]) + 'npy'
            npy_data = np.load(npy_path, mmap_mode='c')

            ground_truth = tensor(self.binary[int(ix)], dtype=torch.float32)
            ground_truth = move_to_cuda(ground_truth)

            # Forward
            out = self.predictor.predict_data_prob(npy_data, model)
            out, _ = torch.max(out, 0)

            # Backward
            loss = self.loss_function(out, ground_truth)
            losses.append(float(loss))

            est_array.append(out.detach().cpu().numpy())
            gt_array.append(ground_truth.detach().cpu().numpy())
        mean_loss = np.mean(losses)
        roc_auc, pr_auc, f1_score = get_metrics(np.array(est_array), np.array(gt_array))
        print(f"[{current_time()}] Loss/Valid: {mean_loss:.4f}")
        print(f"[{current_time()}] F1 Score: {f1_score:.4f}")
        print(f"[{current_time()}] AUC/ROC: {roc_auc:.4f}")
        print(f"[{current_time()}] AUC/PR: {pr_auc:.4f}")
        return Statistics(roc_auc, pr_auc, mean_loss, f1_score)
