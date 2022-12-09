import datetime
import os
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import src.models.model as Model
from src.models.utilites import get_auc


class Solver(object):
    def __init__(self, data_loader, config):
        # data loader
        self.data_loader = data_loader
        self.dataset = config.dataset
        self.data_path = config.data_path

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard

        # model path and step size
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.input_length = config.input_length

        # cuda
        self.is_cuda = torch.cuda.is_available()
        print(f"CUDA: {self.is_cuda}")

        # Build model
        self.valid_list = np.load('split/mtat/valid.npy')
        self.binary = np.load('split/mtat/binary.npy')
        self.build_model()

        # Tensorboard
        self.writer = SummaryWriter()

    def get_model(self):
        return Model.Musicnn(dataset=self.dataset)

    def build_model(self):
        # model
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load pretrained model
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)

        # loss function
        self.loss_function = nn.BCELoss

        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

    def load(self, filename):
        s = torch.load(filename)
        if 'spec.mel_scale.fb' in s.keys():
            self.model.spec.mel_scale.fb = s['spec.mel_scale.fb']
        self.model.load_state_dict(s)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def train(self):
        # Start training
        start_t = time.time()
        reconst_loss = self.loss_function
        best_metric = 0

        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            self.model = self.model.train()
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = self.to_var(x)
                y = self.to_var(y)
                out = self.model(x)

                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # validation
            best_metric = self.validation(best_metric, epoch)

        print("[%s] Train finished. Elapsed: %s"
              % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 datetime.timedelta(seconds=time.time() - start_t)))

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

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

    def print_log(self, epoch, ctr, loss, start_t):
        if ctr % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                  (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   epoch + 1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                   datetime.timedelta(seconds=time.time() - start_t)))

    def validation(self, best_metric, epoch):
        roc_auc, pr_auc, loss = self.get_validation_score(epoch)
        score = 1 - loss
        if score > best_metric:
            print('best model!')
            best_metric = score
            torch.save(self.model.state_dict(),
                       os.path.join(self.model_save_path, 'best_model.pth'))
        return best_metric

    def get_validation_score(self, epoch):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = self.loss_function()
        index = 0
        for line in tqdm.tqdm(self.valid_list):
            ix, fn = line.split('\t')

            # load and split
            x = self.get_tensor(fn)

            # ground truth
            ground_truth = self.binary[int(ix)]

            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for _ in range(self.batch_size)]).cuda()
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)

            gt_array.append(ground_truth)
            index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print('loss: %.4f' % loss)

        roc_auc, pr_auc = get_auc(est_array, gt_array)
        self.writer.add_scalar('Loss/valid', loss, epoch)
        self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', pr_auc, epoch)
        return roc_auc, pr_auc, loss
