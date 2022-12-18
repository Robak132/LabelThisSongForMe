import os
import numpy as np
from torch.utils import data


class AudioFolder(data.Dataset):
    def __init__(self, root, train_path, binary_path, input_length=None):
        self.root = root
        self.input_length = input_length
        self.fl = np.load(train_path, allow_pickle=True)
        self.binary = np.load(binary_path)

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype('float32'), tag_binary.astype('float32')

    def get_npy(self, index):
        ix, fn = self.fl[index]
        npy_path = os.path.join(self.root, 'mtat', 'npy', fn.split('/')[0], fn.split('/')[1][:-3]) + 'npy'
        npy = np.load(npy_path, mmap_mode='c')
        random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
        npy = np.array(npy[random_idx:random_idx + self.input_length])
        tag_binary = self.binary[int(ix)]
        return npy, tag_binary

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
