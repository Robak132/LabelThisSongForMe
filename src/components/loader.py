import os
from pathlib import Path

import numpy as np
from torch import tensor, Tensor
from torch.utils import data


class AudioFolder(data.Dataset):
    def __init__(self, data_path, files_path, binary_path, input_length):
        self.data_path = data_path
        self.input_length = input_length
        self.files = np.load(files_path, allow_pickle=True)
        self.binary = {row[0]: row[1:] for row in np.load(binary_path, allow_pickle=True)}

    def get_random_data_chunk(self, data) -> Tensor:
        random_idx = int(np.floor(np.random.random(1) * (len(data) - self.input_length)))
        return tensor(np.array(data[random_idx:random_idx + self.input_length]))

    def __getitem__(self, index):
        ix, fn = self.files[index]
        npy_path = os.path.join(self.data_path, 'mtat/npy', Path(fn).with_suffix(".npy"))
        data_chunk = self.get_random_data_chunk(np.load(npy_path, mmap_mode='c'))
        return data_chunk, self.binary[int(ix)].astype('float32')

    def __len__(self):
        return len(self.files)


def get_audio_loader(data_path, batch_size, files_path, binary_path, input_length, shuffle=True):
    dataset = AudioFolder(data_path=data_path,
                          files_path=files_path,
                          binary_path=binary_path,
                          input_length=input_length)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=False)
    return data_loader
