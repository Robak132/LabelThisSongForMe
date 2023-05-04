import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import tensor, nn
from tqdm import tqdm

from src.utils.common import move_to_cuda, Statistics, load_model
from src.utils.config import Config
from src.components.predictor import Predictor, SklearnPredictor
from src.interfaces.base_tester import BaseTester


class Tester(BaseTester):
    def __init__(self, config: Config = None, model_filename: str = None, cuda: bool = None, mode: str = "TEST"):
        super().__init__(Predictor(config, cuda, model_filename), config, model_filename, cuda, mode)
        self.loss_function = nn.BCELoss()

    def _load_model(self, model):
        return load_model(os.path.join(self.model_filename_path, self.model_filename), model)

    def test(self, model=None) -> Statistics:
        if model is None:
            model = load_model(os.path.join(self.model_filename_path, self.model_filename), self.model)

        est_array = []
        gt_array = []
        losses = []
        model.eval()
        for ix, mp3_path in tqdm(self.test_list):
            npy_path = os.path.join(self.data_path, 'mtat/npy', Path(mp3_path).with_suffix('.npy'))
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
        return Statistics(np.array(est_array), np.array(gt_array), mean_loss)

class SklearnTester(BaseTester):
    def __init__(self, config: Config = None, model_filename: str = None, mode: str = "TEST"):
        super().__init__(SklearnPredictor(config, model_filename), config, model_filename, False, mode)

    def _load_model(self, model):
        return pickle.load(open(os.path.join(self.model_filename_path, self.model_filename), "rb"))

    def test(self, model=None) -> Statistics:
        if model is None:
            model = self._load_model(os.path.join(self.model_filename_path, self.model_filename))

        gt_array = []
        npy_data_array = []
        for ix, mp3_path in tqdm(self.test_list):
            npy_path = os.path.join(self.data_path, 'mtat/emb', Path(mp3_path).with_suffix('.npy'))
            npy_data_array.append(np.load(npy_path).flatten())
            gt_array.append(self.binary[int(ix)])
        npy_data_array = np.array(npy_data_array)

        est_array = self.predictor.predict_data_prob(npy_data_array, model)
        return Statistics(est_array, np.array(gt_array))
