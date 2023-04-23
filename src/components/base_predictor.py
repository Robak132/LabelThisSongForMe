import os

import numpy as np
import pandas as pd
import torch

from src.components.common import move_to_cuda, Config


class BasePredictor:
    def __init__(self, config: Config, model_filename: str = None, cuda: bool = None):
        if config is None:
            config = Config()
        self.sr = config.sr

        tags_path = os.path.join(config.dataset_split_path, config.dataset_name, "tags.npy")
        self.tags = np.load(tags_path, allow_pickle=True)

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.__class__.__name__)
        self.model_filename = model_filename
        self.data_path = config.data_path
        self.input_length = config.input_length

        # cuda
        self.is_cuda = torch.cuda.is_available() if cuda is None else cuda

        # preprocessor
        self.preprocessor = None

        # model
        try:
            self.model = move_to_cuda(config.model)
        except Exception:
            self.model = config.model

    def predict_tags_prob(self, mp3_file, model=None) -> pd.DataFrame:
        if model is None:
            model = self._load_model()

        out = self.predict_file_prob(mp3_file, model).T
        df = pd.DataFrame(out, index=self.tags)
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
        return df

    def predict_tags(self, mp3_file, model=None) -> list[str]:
        if model is None:
            model = self._load_model()

        out = self.predict_file(mp3_file, model).T
        df = pd.DataFrame(out, index=self.tags)
        df = df[df[0] == 1]
        return df.index.to_list()

    def predict_file_prob(self, mp3_file, model=None):
        if model is None:
            model = self._load_model()

        return self.predict_data_prob(self._preprocessor_func(mp3_file), model)

    def predict_file(self, mp3_file, model=None):
        if model is None:
            model = self._load_model()

        return self.predict_data(self._preprocessor_func(mp3_file), model)

    def predict_data(self, x, model=None):
        raise Exception("This is abstract method!")

    def predict_data_prob(self, x, model=None):
        raise Exception("This is abstract method!")

    def _load_model(self):
        raise Exception("This is abstract method!")

    def _preprocessor_func(self, mp3_file):
        raise Exception("This is abstract method!")
