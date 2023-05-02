import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta

from src.utils.config import Config
from src.utils.common import move_to_cuda, current_time, load_file_lists
from src.utils.loader import get_audio_loader
from src.components.tester import Tester, Statistics


class Trainer:
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        train_path = os.path.join(config.dataset_split_path, config.dataset_name, "train.npy")
        valid_path = os.path.join(config.dataset_split_path, config.dataset_name, "valid.npy")
        test_path = os.path.join(config.dataset_split_path, config.dataset_name, "test.npy")
        binary_path = os.path.join(config.dataset_split_path, config.dataset_name, "binary.npy")

        # run preprocessor if needed
        if config.preprocessor is not None:
            files = load_file_lists([train_path, valid_path, test_path])[:, 1]
            config.preprocessor.run(files)

        # data loader
        self.data_loader = get_audio_loader(data_path=config.data_path,
                                            batch_size=config.batch_size,
                                            files_path=train_path,
                                            binary_path=binary_path,
                                            input_length=config.input_length)
        # training settings
        self.n_epochs = config.n_epochs
        self.log_step = config.log_step

        # model path and step size
        self.dataset_name = config.dataset_name
        self.model_filename_path = os.path.join(config.model_filename_path, config.model.__class__.__name__, self.dataset_name)
        os.makedirs(os.path.join(*self.model_filename_path.split("/")), exist_ok=True)

        # cuda
        self.is_cuda = torch.cuda.is_available()
        print(f"[{current_time()}] Trainer initialised with CUDA: {self.is_cuda}")

        # model
        self.model = move_to_cuda(config.model)
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr, weight_decay=config.weight_decay)

        # Tensorboard
        start_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.model_file_name = f"{start_datetime}.pth"
        self.writer = SummaryWriter(os.path.join(config.logs_path, config.dataset_name, start_datetime))

        # Validator
        self.validator = Tester(config, mode="VALID")

    def train(self):
        # Start training
        start_t = time.time()
        best_metric = 0
        for epoch in range(self.n_epochs):
            # Training
            loss = self.get_training_loss(epoch, start_t)
            self.writer.add_scalar('Loss/train', loss, epoch)

            # Validation
            best_metric = self.validation(best_metric, epoch)

        print(f"{current_time()}] Train finished. Elapsed: {timedelta(seconds=time.time() - start_t)}")

    def get_training_loss(self, epoch, start_t):
        ctr = 0
        avg_loss = 0
        self.model.train()
        for x, y in self.data_loader:
            ctr += 1
            x = move_to_cuda(x)
            y = move_to_cuda(y)

            self.optimizer.zero_grad()

            # Forward
            out = self.model(x)

            # Backward
            loss = self.loss_function(out, y)
            avg_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            # Log
            if ctr % self.log_step == 0:
                self.print_training_log(epoch, ctr, avg_loss, start_t)
        return avg_loss / ctr

    def print_training_log(self, epoch, ctr, avg_loss, start_t):
        print(f"[{current_time()}] "
              f"Epoch [{epoch + 1}/{self.n_epochs}] "
              f"Iter [{ctr}/{len(self.data_loader)}] "
              f"Loss/train: {avg_loss/ctr:.4f} "
              f"Elapsed: {timedelta(seconds=time.time() - start_t)}")

    def add_to_writer(self, stats: Statistics, epoch: int):
        self.writer.add_scalar('Loss/valid', stats.mean_loss, epoch)
        self.writer.add_scalar('AUC/ROC', stats.roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', stats.pr_auc, epoch)
        self.writer.add_scalar('F1 Score', stats.f1_score, epoch)

    def validation(self, best_metric, epoch: int):
        stats = self.validator.test(self.model)
        self.add_to_writer(stats, epoch)
        if stats.f1_score > best_metric:
            print(f'[{current_time()}] Found new best model. Saving...')
            best_metric = stats.f1_score
            torch.save(self.model.state_dict(), os.path.join(self.model_filename_path, self.model_file_name))
            print(f'[{current_time()}] Saved')
        return best_metric
