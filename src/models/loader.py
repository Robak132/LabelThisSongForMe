import os
import numpy as np
from torch.utils import data

from models.common import get_random_data_chunk


class AudioFolder(data.Dataset):
    def __init__(self, data_path, files_path, binary_path, input_length):
        self.data_path = data_path
        self.input_length = input_length
        self.files = np.load(files_path, allow_pickle=True)
        self.binary = np.load(binary_path, allow_pickle=True)

    def __getitem__(self, index):
        ix, fn = self.files[index]
        npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[0], fn.split('/')[1][:-3]) + 'npy'
        data_chunk = get_random_data_chunk(np.load(npy_path, mmap_mode='c'), self.input_length)
        return data_chunk.astype('float32'), self.binary[int(ix)].astype('float32')

    def __len__(self):
        return len(self.files)


def get_audio_loader(data_path, batch_size, files_path, binary_path, input_length, num_workers=0, shuffle=True):
    dataset = AudioFolder(data_path=data_path,
                          files_path=files_path,
                          binary_path=binary_path,
                          input_length=input_length)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader
