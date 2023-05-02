import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import tensor
from tqdm import tqdm

from src.components.common import move_to_cuda, Statistics, current_time, load_model
from src.components.config import Config
from src.components.predictor import Predictor, SklearnPredictor


class BaseTester:
    def __init__(self, config: Config = None, model_filename: str = None, cuda: bool = None, mode: str = "TEST"):
        if config is None:
            config = Config()

        valid_path = os.path.join(config.dataset_split_path, config.dataset_name, "valid.npy")
        test_path = os.path.join(config.dataset_split_path, config.dataset_name, "test.npy")
        binary_path = os.path.join(config.dataset_split_path, config.dataset_name, "binary.npy")

        self.predictor = self._get_predictor(config, cuda, model_filename)

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

    @staticmethod
    def _get_predictor(config, cuda, model_filename):
        raise Exception("This is abstract method!")

    def _load_model(self, model):
        raise Exception("This is abstract method!")

    def test(self, model=None) -> Statistics:
        raise Exception("This is abstract method!")

class Tester(BaseTester):
    def __init__(self, config: Config = None, model_filename: str = None, cuda: bool = None, mode: str = "TEST"):
        super().__init__(config, model_filename, cuda, mode)

    def _load_model(self, model):
        return load_model(model, self.model)

    @staticmethod
    def _get_predictor(config, cuda, model_filename):
        return Predictor(config, cuda, model_filename)

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
        return Statistics(np.array(est_array), np.array(gt_array), mean_loss)

class SklearnTester(BaseTester):
    def __init__(self, config: Config = None, model_filename: str = None, mode: str = "TEST"):
        super().__init__(config, model_filename, False, mode)

    def _load_model(self, model):
        return pickle.load(open(os.path.join(self.model_filename_path, self.model_filename), "rb"))

    @staticmethod
    def _get_predictor(config, cuda, model_filename):
        return SklearnPredictor(config, model_filename)

    def test(self, model=None) -> Statistics:
        if model is None:
            model = self._load_model(os.path.join(self.model_filename_path, self.model_filename))

        gt_array = []
        npy_data_array = []
        for ix, mp3_path in tqdm(self.test_list):
            npy_path = os.path.join(self.data_path, 'mtat/emb', Path(mp3_path).with_suffix('.npy'))
            npy_data_array.append(np.load(npy_path))
            gt_array.append(self.binary[int(ix)])
        npy_data_array = np.array(npy_data_array)

        est_array = self.predictor.predict_data_prob(npy_data_array, model)
        return Statistics(est_array, np.array(gt_array))
