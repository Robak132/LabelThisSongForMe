import os

import numpy as np
import torchopenl3
import tqdm
from models.common import convert_mp3_to_npy


class PreProcessor:
    def __init__(self, config):
        self.sr = config.sr
        self.data_path = config.data_path
        self.result_path = os.path.join(self.data_path, 'mtat/npy')
        self.files = self.get_file_paths([config.train_path, config.valid_path, config.test_path])

    def get_file_paths(self, file_lists):
        files = []
        for file in file_lists:
            for filename in np.load(file, allow_pickle=True)[:, 1]:
                files.append(os.path.join(self.data_path, 'mtat/mp3', filename))
        return files

    def run(self):
        for fn in tqdm.tqdm(self.files):
            result_fn = os.path.join(self.result_path, fn.split('/')[-2], fn.split('/')[-1][:-3]+'npy')
            if not os.path.exists(result_fn):
                try:
                    os.makedirs(os.path.join(*result_fn.split("/")[:-1]), exist_ok=True)
                    x = self.process(fn)
                    np.save(open(result_fn, 'wb'), x)
                except RuntimeError and EOFError:
                    # some audio files are broken
                    print(fn)
                    continue

    def process(self, fn):
        return convert_mp3_to_npy(fn, self.sr)


class OpenL3PreProcessor(PreProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.result_path = os.path.join(self.data_path, 'mtat/emb')

    def process(self, fn):
        x = convert_mp3_to_npy(fn, self.sr)
        emb, ts = torchopenl3.get_audio_embedding(x, self.sr, content_type="music", input_repr="linear",
                                                  embedding_size=512, batch_size=10, sampler="julian")
        return emb.detach().cpu().numpy()
