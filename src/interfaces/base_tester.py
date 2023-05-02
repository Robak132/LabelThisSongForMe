import os

import numpy as np
import torch
from torch import nn

from src.utils.common import Statistics, move_to_cuda, current_time
from src.utils.config import Config


class BaseTester:
    def __init__(self, config: Config = None, model_filename: str = None, cuda: bool = None, mode: str = "TEST"):
        if config is None:
            config = Config()

        valid_path = os.path.join(config.dataset_split_path, config.dataset_name, "valid.npy")
        test_path = os.path.join(config.dataset_split_path, config.dataset_name, "test.npy")
        binary_path = os.path.join(config.dataset_split_path, config.dataset_name, "binary.npy")

        self.predictor = self._get_predictor(config, cuda, model_filename)

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.__class__.__name__,
                                                config.dataset_name)
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
