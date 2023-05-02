import os
import pickle

import numpy as np
from numpy import ndarray
from torch import tensor, Tensor

from src.components.base_predictor import BasePredictor
from src.components.common import move_to_cuda, load_model, convert_mp3_to_npy
from src.components.config import Config
from src.components.preprocessor import OpenL3PreProcessor


class Predictor(BasePredictor):
    def __init__(self, config: Config, model_filename: str = None, cuda: bool = None):
        super().__init__(config, model_filename, cuda)

    def get_data_chunked(self, data) -> Tensor:
        batch_size = len(data) // self.input_length
        return tensor(np.array([data[self.input_length * i: self.input_length * (i + 1)] for i in range(batch_size)]))

    def predict_file_prob(self, mp3_file):
        return self.predict_data_prob(self._preprocessor_func(mp3_file).flatten()).detach().cpu().numpy()

    def predict_data_prob(self, x, model=None):
        if model is None:
            model = self.model

        x = self.get_data_chunked(x)
        x = move_to_cuda(x)
        out = model(x)
        return out

    def _load_model(self, model):
        return load_model(os.path.join(self.model_filename_path, self.model_filename), model)

    def _preprocessor_func(self, mp3_file) -> ndarray:
        return convert_mp3_to_npy(mp3_file, self.sr)


class SklearnPredictor(BasePredictor):
    def __init__(self, config: Config, model_filename: str = None, cuda: bool = None):
        super().__init__(config, model_filename, cuda)
        self.preprocessor = OpenL3PreProcessor()

    def predict_data_prob(self, x, model=None):
        if model is None:
            model = self.model

        out = model.predict_proba(x.reshape(-1, 15360))
        return np.array(out)[:, :, 1].T

    def _load_model(self, model):
        return pickle.load(open(os.path.join(self.model_filename_path, self.model_filename), "rb"))

    def _preprocessor_func(self, mp3_file) -> ndarray:
        return self.preprocessor.process(mp3_file)
