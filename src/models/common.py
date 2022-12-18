import os

import numpy as np
import torch
from sklearn import metrics
import tqdm


def get_auc(est_array, gt_array):
    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    return roc_aucs, pr_aucs


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def load_tensor_chunked(data_path, file_path, input_length, batch_size):
    # load audio
    npy_path = os.path.join(data_path, 'mtat', 'npy', file_path.split('/')[0], file_path.split('/')[1][:-3]) + 'npy'
    raw = np.load(npy_path, mmap_mode='c')

    # split chunk
    length = len(raw)
    hop = (length - input_length) // batch_size
    x = torch.zeros(batch_size, input_length)
    for i in range(batch_size):
        x[i] = torch.tensor(raw[i * hop:i * hop + input_length]).unsqueeze(0)
    return x


def get_test_score(loss_function, test_list, data_path, input_length, batch_size, model, binary):
    est_array = []
    gt_array = []
    losses = []
    for ix, fn in tqdm.tqdm(test_list):
        x = load_tensor_chunked(data_path, fn, input_length, batch_size)
        ground_truth = binary[int(ix)]

        # forward
        x = to_var(x)
        y = torch.tensor([ground_truth.astype('float32') for _ in range(batch_size)]).cuda()
        out = model(x)
        loss = loss_function(out, y)
        losses.append(float(loss))
        out = out.detach().cpu()

        # estimate
        estimated = np.array(out).mean(axis=0)
        est_array.append(estimated)
        gt_array.append(ground_truth)

    loss = np.mean(losses)
    roc_auc, pr_auc = get_auc(np.array(est_array), np.array(gt_array))
    return roc_auc, pr_auc, loss
