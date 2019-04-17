#!/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

"""
Train Timeception layers on different datasets. There are two different ways to train Timeception.
 1. Timeception-only (TCO): only timeception layers are trained, using features extracted from backbone CNNs.
 2. End-to-end (ETE): timeception is trained on top of backbone CNN. The input is video frames passed throughtout the backboneCNN
    and then the resulted feature is fed to Timeception layers. Here, you enjoy all the benefits of end-to-end training.
    For example, do pre-processing to the input frames, randomly sample the frames, temporal jittering, ...., etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import logging
import datetime
import numpy as np
from optparse import OptionParser

import torch
import torch.utils.data

from torch.nn import functional as F
from torch.nn import Module, Dropout, BatchNorm1d, LeakyReLU, Linear, LogSoftmax, Sigmoid
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchviz
import torchvision
import torchsummary

from nets import timeception_pytorch
from core import utils, pytorch_utils, image_utils, config_utils, const, config, data_utils, metrics
from core.utils import Path as Pth

logger = logging.getLogger(__name__)

def train_tco():
    """
    Train Timeception layers based on the given configurations.
    This train scheme is Timeception-only (TCO).
    """

    # get some configs for the training
    n_epochs = config.cfg.TRAIN.N_EPOCHS
    dataset_name = config.cfg.DATASET_NAME
    model_name = '%s_%s' % (config.cfg.MODEL.NAME, utils.timestamp())
    device = 'cuda'

    # data generators
    loader_tr, n_samples_tr, n_batches_tr = __define_loader(is_training=True)
    loader_te, n_samples_te, n_batches_te = __define_loader(is_training=False)

    logger.info('--- start time')
    logger.info(datetime.datetime.now())
    logger.info('... [tr]: n_samples, n_batch, batch_size: %d, %d, %d' % (n_samples_tr, n_batches_tr, config.cfg.TRAIN.BATCH_SIZE))
    logger.info('... [te]: n_samples, n_batch, batch_size: %d, %d, %d' % (n_samples_te, n_batches_te, config.cfg.TEST.BATCH_SIZE))

    # load model
    model, optimizer, loss_fn, metric_fn, metric_fn_name = __define_timeception_model(device)
    logger.info(pytorch_utils.summary(model, model._input_shape[1:], batch_size=2, device='cuda'))

    # save the model
    model_saver = pytorch_utils.ModelSaver(model, dataset_name, model_name)

    # loop on the epochs
    sys.stdout.write('\n')
    for idx_epoch in range(n_epochs):

        epoch_num = idx_epoch + 1

        loss_tr = 0.0
        acc_tr = 0.0
        loss_te = 0.0
        acc_te = 0.0

        tt1 = time.time()

        # flag model as training
        model.train()

        # training
        for idx_batch, (x, y_true) in enumerate(loader_tr):
            batch_num = idx_batch + 1

            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            y_true = y_true.cpu().numpy().astype(np.int32)
            y_pred = y_pred.cpu().detach().numpy()
            loss_b_tr = loss.cpu().detach().numpy()
            acc_b_tr = metric_fn(y_true, y_pred)

            loss_tr += loss_b_tr
            acc_tr += acc_b_tr
            loss_b_tr = loss_tr / float(batch_num)
            acc_b_tr = acc_tr / float(batch_num)
            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds - epoch: %02d/%02d, batch [tr]: %02d/%02d, loss, %s: %0.2f, %0.2f ' % (duration, epoch_num, n_epochs, batch_num, n_batches_tr, metric_fn_name, loss_b_tr, acc_b_tr))

        # flag model as testing
        model.eval()

        # testing
        for idx_batch, (x, y_true) in enumerate(loader_te):
            batch_num = idx_batch + 1

            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            loss_b_te = loss_fn(y_pred, y_true).cpu().detach().numpy()
            y_true = y_true.cpu().numpy().astype(np.int32)
            y_pred = y_pred.cpu().detach().numpy()
            acc_b_te = metric_fn(y_true, y_pred)

            loss_te += loss_b_te
            acc_te += acc_b_te
            loss_b_te = loss_te / float(batch_num)
            acc_b_te = acc_te / float(batch_num)
            tt2 = time.time()
            duration = tt2 - tt1
            sys.stdout.write('\r%04ds - epoch: %02d/%02d, batch [te]: %02d/%02d, loss, %s: %0.2f, %0.2f ' % (duration, epoch_num, n_epochs, batch_num, n_batches_te, metric_fn_name, loss_b_te, acc_b_te))

        loss_tr /= float(n_batches_tr)
        loss_te /= float(n_batches_te)
        acc_tr /= float(n_batches_tr)
        acc_te /= float(n_batches_te)

        tt2 = time.time()
        duration = tt2 - tt1
        sys.stdout.write('\r%04ds - epoch: %02d/%02d, [tr]: %0.2f, %0.2f, [te]: %0.2f, %0.2f           \n' % (duration, epoch_num, n_epochs, loss_tr, acc_te, loss_te, acc_te))

        # after each epoch, save data
        model_saver.save(idx_epoch)

    logger.info('--- finish time')
    logger.info(datetime.datetime.now())

def train_ete():
    """
    Train Timeception layers based on the given configurations.
    This train scheme is End-to-end (ETE).
    """

    raise Exception('Sorry, not implemented yet!')

def __define_loader(is_training):
    """
    Define data loader.
    """

    # get some configs for the training
    n_classes = config.cfg.MODEL.N_CLASSES
    dataset_name = config.cfg.DATASET_NAME
    backbone_model_name = config.cfg.MODEL.BACKBONE_CNN
    backbone_feature_name = config.cfg.MODEL.BACKBONE_FEATURE
    n_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS
    n_workers = config.cfg.TRAIN.N_WORKERS

    batch_size_tr = config.cfg.TRAIN.BATCH_SIZE
    batch_size_te = config.cfg.TEST.BATCH_SIZE
    batch_size = batch_size_tr if is_training else batch_size_te

    # size and name of feature
    feature_name = 'features_%s_%s_%sf' % (backbone_model_name, backbone_feature_name, n_timesteps)
    c, h, w = utils.get_model_feat_maps_info(backbone_model_name, backbone_feature_name)
    feature_dim = (c, n_timesteps, h, w)

    # data generators
    params = {'batch_size': batch_size, 'n_classes': n_classes, 'feature_name': feature_name, 'feature_dim': feature_dim, 'is_training': is_training}
    dataset_class = data_utils.PYTORCH_DATASETS_DICT[dataset_name]
    dataset = dataset_class(**params)
    n_samples = dataset.n_samples
    n_batches = dataset.n_batches

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    return data_loader, n_samples, n_batches

def __define_timeception_model(device):
    """
    Define model, optimizer, loss function and metric function.
    """
    # some configurations
    classification_type = config.cfg.MODEL.CLASSIFICATION_TYPE
    solver_name = config.cfg.SOLVER.NAME
    solver_lr = config.cfg.SOLVER.LR
    adam_epsilon = config.cfg.SOLVER.ADAM_EPSILON

    # define model
    model = Model().to(device)
    model_param = model.parameters()

    # define the optimizer
    optimizer = SGD(model_param, lr=0.01) if solver_name == 'sgd' else Adam(model_param, lr=solver_lr, eps=adam_epsilon)

    # loss and evaluation function for either multi-label "ml" or single-label "sl" classification
    if classification_type == 'ml':
        loss_fn = torch.nn.BCELoss()
        metric_fn = metrics.map_charades
        metric_fn_name = 'map'
    else:
        loss_fn = torch.nn.NLLLoss()
        metric_fn = metrics.accuracy
        metric_fn_name = 'acc'

    return model, optimizer, loss_fn, metric_fn, metric_fn_name

class Model(Module):
    """
    Define Timeception classifier.
    """

    def __init__(self):
        super(Model, self).__init__()

        # some configurations for the model
        n_tc_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS
        backbone_name = config.cfg.MODEL.BACKBONE_CNN
        feature_name = config.cfg.MODEL.BACKBONE_FEATURE
        n_tc_layers = config.cfg.MODEL.N_TC_LAYERS
        n_classes = config.cfg.MODEL.N_CLASSES
        is_dilated = config.cfg.MODEL.MULTISCALE_TYPE
        OutputActivation = Sigmoid if config.cfg.MODEL.CLASSIFICATION_TYPE == 'ml' else LogSoftmax
        n_channels_in, channel_h, channel_w = utils.get_model_feat_maps_info(backbone_name, feature_name)
        n_groups = int(n_channels_in / 128.0)

        input_shape = (None, n_channels_in, n_tc_timesteps, channel_h, channel_w)  # (C, T, H, W)
        self._input_shape = input_shape

        # define 4 layers of timeception
        self.timeception = timeception_pytorch.Timeception(input_shape, n_tc_layers, n_groups, is_dilated)  # (C, T, H, W)

        # get number of output channels after timeception
        n_channels_in = self.timeception.n_channels_out

        # define layers for classifier
        self.do1 = Dropout(0.5)
        self.l1 = Linear(n_channels_in, 512)
        self.bn1 = BatchNorm1d(512)
        self.ac1 = LeakyReLU(0.2)
        self.do2 = Dropout(0.25)
        self.l2 = Linear(512, n_classes)
        self.ac2 = OutputActivation()

    def forward(self, input):
        # feedforward the input to the timeception layers
        tensor = self.timeception(input)

        # max-pool over space-time
        bn, c, t, h, w = tensor.size()
        tensor = tensor.view(bn, c, t * h * w)
        tensor = torch.max(tensor, dim=2, keepdim=False)
        tensor = tensor[0]

        # dense layers for classification
        tensor = self.do1(tensor)
        tensor = self.l1(tensor)
        tensor = self.bn1(tensor)
        tensor = self.ac1(tensor)
        tensor = self.do2(tensor)
        tensor = self.l2(tensor)
        tensor = self.ac2(tensor)

        return tensor

def __main():
    """
    Run this script to train Timeception.
    """

    default_config_file = 'charades_i3d_tc4_f1024.yaml'
    default_config_file = 'charades_i3d_tc2_f256.yaml'

    # Parse the arguments
    parser = OptionParser()
    parser.add_option('-c', '--config_file', dest='config_file', default=default_config_file, help='Yaml config file that contains all training details.')
    (options, args) = parser.parse_args()
    config_file = options.config_file

    # check if exist
    if config_file is None or config_file == '':
        msg = 'Config file not passed, default config is used: %s' % (config_file)
        logging.warning(msg)
        config_file = default_config_file

    # path of config file
    config_path = './configs/%s' % (config_file)

    # check if file exist
    if not os.path.exists(config_path):
        msg = 'Sorry, could not find config file with the following path: %s' % (config_path)
        logging.error(msg)
    else:
        # read the config from file and copy it to the project configuration "cfg"
        config_utils.cfg_from_file(config_path)

        # choose which training scheme, either 'ete' or 'tco'
        training_scheme = config.cfg.TRAIN.SCHEME

        # start training
        if training_scheme == 'tco':
            train_tco()
        else:
            train_ete()

if __name__ == '__main__':
    __main()
