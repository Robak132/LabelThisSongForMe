import os

import numpy as np
import pandas as pd
import torch

from src.utils.config import Config
from src.utils.common import move_to_cuda


class BasePredictor:
    def __init__(self, config: Config, model_filename: str = None, cuda: bool = None):
        if config is None:
            config = Config()
        self.sr = config.sr

        tags_path = os.path.join(config.dataset_split_path, config.dataset_name, "tags.npy")
        self.tags = np.load(tags_path, allow_pickle=True)

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.__class__.__name__, config.dataset_name)
        self.model_filename = model_filename
        self.data_path = config.data_path
        self.input_length = config.input_length

        # cuda
        self.is_cuda = torch.cuda.is_available() if cuda is None else cuda

        # preprocessor
        self.preprocessor = None

        # model
        if model_filename is not None:
            self.model = self._load_model(config.model)
            self.model = move_to_cuda(self.model)

    def predict_tags_prob(self, mp3_file) -> pd.DataFrame:
        out = self.predict_file_prob(mp3_file).T
        df = pd.DataFrame(out, index=self.tags)
        return df

    def predict_file_prob(self, mp3_file):
        return self.predict_data_prob(self._preprocessor_func(mp3_file).flatten())

    def predict_data_prob(self, x, model=None):
        raise Exception("This is abstract method!")

    def _preprocessor_func(self, mp3_file):
        raise Exception("This is abstract method!")
