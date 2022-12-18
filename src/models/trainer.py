import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.models.common import to_var, get_test_score
from src.models.loader import get_audio_loader


class Trainer:
    def __init__(self, config):
        # create folders if they don't exist
        os.makedirs(os.path.join(*config.model_save_path.split("/")[:-1]), exist_ok=True)

        # data loader
        self.data_loader = get_audio_loader(config.data_path,
                                            batch_size=config.batch_size,
                                            train_path=config.train_path,
                                            binary_path=config.binary_path,
                                            num_workers=config.num_workers,
                                            input_length=config.input_length)
        self.data_path = config.data_path

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr

        # model path and step size
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.input_length = config.input_length

        # cuda
        self.is_cuda = torch.cuda.is_available()
        print(f"CUDA: {self.is_cuda}")

        # Build model
        self.valid_list = np.load(config.valid_path, allow_pickle=True)
        self.binary = np.load(config.binary_path, allow_pickle=True)
        self.build_model(config.model)

        # Tensorboard
        self.writer = SummaryWriter()

    def build_model(self, model):
        # model
        self.model = model

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # loss function
        self.loss_function = nn.BCELoss()

        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

    def load(self, filename):
        s = torch.load(filename)
        if 'spec.mel_scale.fb' in s.keys():
            self.model.spec.mel_scale.fb = s['spec.mel_scale.fb']
        self.model.load_state_dict(s)

    def train(self):
        # Start training
        start_t = time.time()
        best_metric = 0
        reconstruction_loss = self.loss_function

        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            loss = None
            self.model = self.model.train()

            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = to_var(x)
                y = to_var(y)
                out = self.model(x)

                # Backward
                loss = reconstruction_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)

            if loss is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # validation
            best_metric = self.validation(best_metric, epoch)

        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train finished. "
              f"Elapsed: {datetime.timedelta(seconds=time.time() - start_t)}")

    def print_log(self, epoch, ctr, loss, start_t):
        if ctr % self.log_step == 0:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Epoch [{epoch + 1}/{self.n_epochs}] Iter [{ctr}/{len(self.data_loader)}] "
                  f"train loss: {loss.item():.4f} Elapsed: {datetime.timedelta(seconds=time.time() - start_t)}")

    def validation(self, best_metric, epoch):
        self.model = self.model.eval()
        roc_auc, pr_auc, loss = get_test_score(self.loss_function,
                                               self.valid_list,
                                               self.data_path,
                                               self.input_length,
                                               self.batch_size,
                                               self.model,
                                               self.binary)
        print(f'loss: {loss:.4f}')
        print(f'roc_auc: {roc_auc:.4f}')
        print(f'pr_auc: {pr_auc:.4f}')
        self.writer.add_scalar('Loss/valid', loss, epoch)
        self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', pr_auc, epoch)

        score = 1 - loss
        if score > best_metric:
            print('Found new best model')
            best_metric = score
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path))
        return best_metric
