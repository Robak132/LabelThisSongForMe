import os
import pickle

import numpy as np
import pandas as pd
import torch

from src.components.preprocessor import OpenL3PreProcessor
from src.components.common import move_to_cuda, load_model, Config, convert_mp3_to_npy, get_data_chunked


class Predictor:
    def __init__(self, config: Config, model_filename: str = None, cuda: bool = None):
        if config is None:
            config = Config()
        self.sr = config.sr

        tags_path = os.path.join(config.dataset_split_path, config.dataset_name, "tags.npy")
        self.tags = np.load(tags_path, allow_pickle=True)

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.__class__.__name__)
        self.model_filename = model_filename

        # cuda
        self.is_cuda = torch.cuda.is_available() if cuda is None else cuda

        # model
        self.model = move_to_cuda(config.model)

        self.data_path = config.data_path
        self.input_length = config.input_length

    def predict_tags(self, mp3_file, model=None) -> pd.DataFrame:
        if model is None:
            model = load_model(os.path.join(self.model_filename_path, self.model_filename), self.model)

        out = self.predict_file(mp3_file, model).T
        df = pd.DataFrame(out, index=self.tags)
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
        return df

    def predict_file(self, mp3_file, model=None):
        if model is None:
            model = load_model(os.path.join(self.model_filename_path, self.model_filename), self.model)

        return self.predict_data(convert_mp3_to_npy(mp3_file, self.sr), model)

    def predict_data(self, x, model=None):
        if model is None:
            model = load_model(self.model_filename_path, self.model)

        x = get_data_chunked(x, self.input_length)
        x = move_to_cuda(x)
        out = model(x).detach().cpu()
        return out

class SklearnPredictor:
    def __init__(self, config: Config, model_filename: str = None, cuda: bool = None):
        if config is None:
            config = Config()

        tags_path = os.path.join(config.dataset_split_path, config.dataset_name, "tags.npy")
        self.tags = np.load(tags_path, allow_pickle=True)

        self.model_filename_path = config.model_filename_path
        self.model_filename = model_filename

        # cuda
        self.is_cuda = torch.cuda.is_available() if cuda is None else cuda

        self.data_path = config.data_path
        self.input_length = config.input_length

        self.preprocessor = OpenL3PreProcessor()

    def predict_tags(self, mp3_file, model=None) -> pd.DataFrame:
        if model is None:
            model = pickle.load(open(os.path.join(self.model_filename_path, self.model_filename), "rb"))

        out = self.predict_file(mp3_file, model).T
        df = pd.DataFrame(out, index=self.tags)
        df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
        return df

    def predict_file(self, mp3_file, model=None):
        if model is None:
            model = pickle.load(open(os.path.join(self.model_filename_path, self.model_filename), "rb"))

        return self.predict_data(self.preprocessor.process(mp3_file), model)

    def predict_data(self, x, model=None):
        if model is None:
            model = pickle.load(open(os.path.join(self.model_filename_path, self.model_filename), "rb"))

        return model.predict(x.reshape(1, -1))
