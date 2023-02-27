import numpy as np
import torch
import torch.nn as nn

from models.common import get_test_score


class Tester:
    def __init__(self, config):
        self.model_save_path = config.model_save_path
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.input_length = config.input_length

        self.is_cuda = torch.cuda.is_available()
        self.build_model(config.model)

        self.test_list = np.load(config.test_path, allow_pickle=True)
        self.binary = np.load(config.binary_path, allow_pickle=True)

    def build_model(self, model):
        self.model = model

        # load model
        self.load_model(self.model_save_path)

        # loss function
        self.loss_function = nn.BCELoss()

        # cuda
        if self.is_cuda:
            self.model.cuda()

    def load_model(self, filename):
        s = torch.load(filename)
        if 'spec.mel_scale.fb' in s.keys():
            self.model.spec.mel_scale.fb = s['spec.mel_scale.fb']
        self.model.load_state_dict(s)

    def test(self):
        self.model = self.model.eval()
        roc_auc, pr_auc, loss = get_test_score(self.loss_function,
                                               self.test_list,
                                               self.data_path,
                                               self.input_length,
                                               self.batch_size,
                                               self.model,
                                               self.binary)
        print(f'loss: {loss:.4f}')
        print(f'roc_auc: {roc_auc:.4f}')
        print(f'pr_auc: {pr_auc:.4f}')