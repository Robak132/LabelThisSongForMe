import os
import warnings
from glob import glob

import librosa
import numpy as np
import tqdm

warnings.filterwarnings('ignore')


class PreProcessor:
    def __init__(self, data_path):
        self.fs = 16000
        self.data_path = data_path

    def get_file_paths(self, *args):
        files = []
        if len(args) != 0:
            for arg in args:
                files += list(os.path.join(self.data_path, 'mtat', 'mp3', filename) for filename in np.load(arg, allow_pickle=True)[:, 1])
        else:
            files = glob(os.path.join(self.data_path, 'mtat', 'mp3', '*/*.mp3'))
        return files

    def get_npy(self, fn):
        x, sr = librosa.core.load(fn, sr=self.fs)
        return x

    def run(self, *args):
        files = self.get_file_paths(*args)
        self.npy_path = os.path.join(self.data_path, 'mtat', 'npy')

        for fn in tqdm.tqdm(files):
            npy_fn = os.path.join(self.npy_path, fn.split('/')[-2], fn.split('/')[-1][:-3]+'npy')
            if not os.path.exists(npy_fn):
                try:
                    os.makedirs(npy_fn[:-1], exist_ok=True)
                    x = self.get_npy(fn)
                    np.save(open(npy_fn, 'wb'), x)
                except RuntimeError and EOFError:
                    # some audio files are broken
                    print(fn)
                    continue


if __name__ == '__main__':
    p = PreProcessor("data")
    p.run("split/mtat-mini/train.npy", "split/mtat-mini/valid.npy", "split/mtat-mini/test.npy")
