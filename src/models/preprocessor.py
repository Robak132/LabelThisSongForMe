import os
import warnings

import numpy as np
import tqdm

from models.common import convert_mp3_to_npy

warnings.simplefilter(action='ignore', category=UserWarning)


class PreProcessor:
    def __init__(self, config):
        self.sr = config.sr
        self.data_path = config.data_path
        self.files = self.get_file_paths([config.train_path, config.valid_path, config.test_path])

    def get_file_paths(self, npy_files):
        files = []
        for npy_file in npy_files:
            for filename in np.load(npy_file, allow_pickle=True)[:, 1]:
                files.append(os.path.join(self.data_path, 'mtat/mp3', filename))
        return files

    def run(self):
        npy_path = os.path.join(self.data_path, 'mtat/npy')

        for fn in tqdm.tqdm(self.files):
            npy_fn = os.path.join(npy_path, fn.split('/')[-2], fn.split('/')[-1][:-3]+'npy')
            if not os.path.exists(npy_fn):
                try:
                    os.makedirs(os.path.join(*npy_fn.split("/")[:-1]), exist_ok=True)
                    x = convert_mp3_to_npy(fn, self.sr)
                    np.save(open(npy_fn, 'wb'), x)
                except RuntimeError and EOFError:
                    # some audio files are broken
                    print(fn)
                    continue
