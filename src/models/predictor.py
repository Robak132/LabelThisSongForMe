import os

import numpy as np
import pandas as pd
import torch

from src.models.common import move_to_cuda, load_model, Config, convert_mp3_to_npy, get_data_chunked


class Predictor:
    def __init__(self, config: Config, model_filename: str = None, cuda: bool = None):
        if config is None:
            config = Config()
        self.sr = config.sr

        tags_path = os.path.join(config.dataset_split_path, config.dataset_name, "tags.npy")
        self.tags = np.load(tags_path, allow_pickle=True)

        self.model_filename_path = os.path.join(config.model_filename_path, config.model.get_name())
        self.model_filename = model_filename

        # cuda
        self.is_cuda = torch.cuda.is_available() if cuda is None else cuda

        # model
        self.model = move_to_cuda(config.model)

        self.data_path = config.data_path
        self.input_length = config.input_length

    def predict_tags(self, mp3_file=None, model=None) -> pd.DataFrame:
        if model is None:
            model = load_model(os.path.join(self.model_filename_path, self.model_filename), self.model)
        if mp3_file is not None:
            out = self.predict_mp3(mp3_file, model).T
            df = pd.DataFrame(out, index=self.tags)
            df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
            return df

    def predict_mp3(self, x, model=None):
        if model is None:
            model = load_model(os.path.join(self.model_filename_path, self.model_filename), self.model)

        return self.predict_npy(convert_mp3_to_npy(x, self.sr), model)

    def predict_npy(self, x, model=None):
        if model is None:
            model = load_model(self.model_filename_path, self.model)

        x = get_data_chunked(x, self.input_length)
        x = move_to_cuda(x)
        out = model(x).detach().cpu()
        return out
