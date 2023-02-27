import librosa
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from external.model import Musicnn
from models.common import to_var, convert_mp3_to_npy


class Provider:
    def __init__(self, model_path):
        self.model = Musicnn(dataset='mtat')
        self.model_path = model_path
        self.is_cuda = torch.cuda.is_available()
        self.tags = np.load("split/mtat/tags.npy", allow_pickle=True)

        self.build_model()

    def build_model(self):
        # Load model
        s = torch.load(self.model_path)
        if 'spec.mel_scale.fb' in s.keys():
            self.model.spec.mel_scale.fb = s['spec.mel_scale.fb']
        self.model.load_state_dict(s)

        # Cuda
        if self.is_cuda:
            self.model.cuda()

    def split_tensor_to_chunks(self, data, input_length, batch_size=None) -> Tensor:
        if batch_size is None:
            batch_size = len(data) // input_length

        hop = (len(data) - input_length) // batch_size
        x = torch.zeros(batch_size, input_length)
        for i in range(batch_size):
            x[i] = torch.tensor(data[i * hop:i * hop + input_length]).unsqueeze(0)
        return x

    def get_tags(self, mp3_file, input_length=3 * 16000):
        self.model.eval()
        raw = convert_mp3_to_npy(mp3_file, sr=16000)
        x = self.split_tensor_to_chunks(raw, input_length)

        x = to_var(x)
        out = self.model(x)
        return out.detach().cpu()
