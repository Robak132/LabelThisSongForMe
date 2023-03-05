import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.tester import Tester, Statistics
from models.common import move_to_cuda, current_time, Config
from models.loader import get_audio_loader


class Trainer:
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        # create folders if they don't exist
        os.makedirs(os.path.join(*config.model_save_path.split("/")[:-1]), exist_ok=True)

        # data loader
        self.data_loader = get_audio_loader(data_path=config.data_path,
                                            batch_size=config.batch_size,
                                            files_path=config.train_path,
                                            binary_path=config.binary_path,
                                            num_workers=config.num_workers,
                                            input_length=config.input_length)
        # training settings
        self.n_epochs = config.n_epochs

        # model path and step size
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step

        # cuda
        self.is_cuda = torch.cuda.is_available()
        print(f"[{current_time()}] Trainer initialised with CUDA: {self.is_cuda}")

        # model
        self.model = move_to_cuda(config.model)
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr, weight_decay=1e-4)

        # Tensorboard
        self.writer = SummaryWriter()

        # Validator
        self.validator = Tester(config, "VALID")

    def train(self):
        # Start training
        start_t = time.time()
        best_metric = 0
        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            loss = None
            self.model.train()
            for x, y in self.data_loader:
                ctr += 1
                x = move_to_cuda(x)
                y = move_to_cuda(y)

                # Forward
                out = self.model(x)

                # Backward
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                if ctr % self.log_step == 0:
                    self.print_training_log(epoch, ctr, loss, start_t)

            if loss is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # Validation
            best_metric = self.validation(best_metric, epoch)

        print(f"{current_time()}] Train finished. Elapsed: {datetime.timedelta(seconds=time.time() - start_t)}")

    def print_training_log(self, epoch, ctr, loss, start_t):
        print(f"[{current_time()}] "
              f"Epoch [{epoch + 1}/{self.n_epochs}] "
              f"Iter [{ctr}/{len(self.data_loader)}] "
              f"Loss/train: {loss.item():.4f} "
              f"Elapsed: {datetime.timedelta(seconds=time.time() - start_t)}")

    def add_to_writer(self, stats: Statistics, epoch: int):
        self.writer.add_scalar('Loss/valid', stats.mean_loss, epoch)
        self.writer.add_scalar('AUC/ROC', stats.roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', stats.pr_auc, epoch)

    def validation(self, best_metric, epoch: int):
        _, stats = self.validator.test(self.model)
        self.add_to_writer(stats, epoch)

        score = 1 - stats.mean_loss
        if score > best_metric:
            print(f'[{current_time()}] Found new best model')
            best_metric = score
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path))
        return best_metric
