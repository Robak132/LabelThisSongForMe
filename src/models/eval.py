# coding: utf-8
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn import metrics
from torch.autograd import Variable

import src.models.model as Model


class Predict(object):
    def __init__(self, config):
        self.model_load_path = config.model_load_path
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()
        self.get_dataset()

    def get_model(self):
        self.input_length = 3 * 16000
        return Model.Musicnn(dataset=self.dataset)

    def build_model(self):
        self.model = self.get_model()

        # load model
        self.load(self.model_load_path)

        # cuda
        if self.is_cuda:
            self.model.cuda()

    def get_dataset(self):
        self.test_list = np.load('split/mtat/test.npy')
        self.binary = np.load('split/mtat/binary.npy')

    def load(self, filename):
        S = torch.load(filename)
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_tensor(self, fn):
        # load audio
        npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
        raw = np.load(npy_path, mmap_mode='r')

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i * hop:i * hop + self.input_length]).unsqueeze(0)
        return x

    def get_auc(self, est_array, gt_array):
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        return roc_aucs, pr_aucs

    def test(self):
        roc_auc, pr_auc, loss = self.get_test_score()
        print('loss: %.4f' % loss)
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.test_list):
            ix, fn = line.split('\t')

            # load and split
            x = self.get_tensor(fn)

            # ground truth
            ground_truth = self.binary[int(ix)]

            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        return roc_auc, pr_auc, loss
