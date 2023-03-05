import os
import numpy as np
from torch.utils import data

from models.common import get_tensor_chunked


class AudioFolder(data.Dataset):
    def __init__(self, root, train_path, binary_path, input_length=None):
        self.root = root
        self.input_length = input_length
        self.fl = np.load(train_path, allow_pickle=True)
        self.binary = np.load(binary_path, allow_pickle=True)

    def __getitem__(self, index):
        ix, fn = self.fl[index]
        npy_path = os.path.join(self.root, 'mtat', 'npy', fn.split('/')[0], fn.split('/')[1][:-3]) + 'npy'
        data_chunk = get_tensor_chunked(np.load(npy_path, mmap_mode='c'), self.input_length)
        return data_chunk.astype('float32'), self.binary[int(ix)].astype('float32')

    def __len__(self):
        return len(self.fl)


def get_audio_loader(root, batch_size, train_path, binary_path, num_workers=0, input_length=None):
    dataset = AudioFolder(root,
                          train_path=train_path,
                          binary_path=binary_path,
                          input_length=input_length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader
