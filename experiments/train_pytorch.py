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

import logging
import os
import datetime
import numpy as np
from optparse import OptionParser

import torch
import torch.utils.data

from torch.nn import functional as F
from torch.nn import Module, Dropout, BatchNorm1d, LeakyReLU, Linear, LogSoftmax
from torch.autograd import Variable
from torchvision import datasets, transforms

import torchviz
import torchvision
import torchsummary

# import tensorflow as tf
# import keras.backend as K
# from keras.layers import Dense, LeakyReLU, Dropout, Input, Activation, BatchNormalization
# from keras.optimizers import SGD, Adam
# from keras.models import Model

from nets import timeception_pytorch
from core import utils, pytorch_utils, image_utils, config_utils, const, config, data_utils
from core.utils import Path as Pth

logger = logging.getLogger(__name__)

def train_tco():
    """
    Train Timeception layers based on the given configurations.
    This train scheme is Timeception-only (TCO).
    """

    # get some configs for the training
    n_workers = config.cfg.TRAIN.N_WORKERS
    n_epochs = config.cfg.TRAIN.N_EPOCHS
    dataset_name = config.cfg.DATASET_NAME
    model_name = '%s_%s' % (config.cfg.MODEL.NAME, utils.timestamp())

    # data generators
    data_generator_tr = __define_data_generator(is_training=True)
    data_generator_te = __define_data_generator(is_training=False)

    logger.info('--- start time')
    logger.info(datetime.datetime.now())
    logger.info('... [tr]: n_samples, n_batch, batch_size: %d, %d, %d' % (data_generator_tr.n_samples, data_generator_tr.n_batches, config.cfg.TRAIN.BATCH_SIZE))
    logger.info('... [te]: n_samples, n_batch, batch_size: %d, %d, %d' % (data_generator_te.n_samples, data_generator_te.n_batches, config.cfg.TEST.BATCH_SIZE))

    # callback to save the model
    # save_callback = keras_utils.SaveCallback(dataset_name, model_name)

    # load model
    model = Model()
    logger.info(pytorch_utils.summary(model, model._input_shape[1:], batch_size=-1, device='cpu'))



    # train the model
    model.fit_generator(epochs=n_epochs, generator=data_generator_tr, validation_data=data_generator_te, use_multiprocessing=True, workers=n_workers, callbacks=[save_callback], verbose=2)

    logger.info('--- finish time')
    logger.info(datetime.datetime.now())

def train_ete():
    """
    Train Timeception layers based on the given configurations.
    This train scheme is End-to-end (ETE).
    """

    raise Exception('Sorry, not implemented yet!')

def __define_data_generator(is_training):
    """
    Define data generator.
    """

    # get some configs for the training
    n_classes = config.cfg.MODEL.N_CLASSES
    dataset_name = config.cfg.DATASET_NAME
    backbone_model_name = config.cfg.MODEL.BACKBONE_CNN
    backbone_feature_name = config.cfg.MODEL.BACKBONE_FEATURE
    n_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS

    batch_size_tr = config.cfg.TRAIN.BATCH_SIZE
    batch_size_te = config.cfg.TEST.BATCH_SIZE
    batch_size = batch_size_tr if is_training else batch_size_te

    # size and name of feature
    feature_name = 'features_%s_%s_%sf' % (backbone_model_name, backbone_feature_name, n_timesteps)
    c, h, w = utils.get_model_feat_maps_info(backbone_model_name, backbone_feature_name)
    feature_dim = (n_timesteps, h, w, c)

    # data generators
    params = {'batch_size': batch_size, 'n_classes': n_classes, 'feature_name': feature_name, 'feature_dim': feature_dim, 'is_shuffle': True, 'is_training': is_training}

    # batch_size, n_channels, n_classes, is_training, shuffle=True
    data_generator_class = data_utils.DATA_GENERATOR_DICT[dataset_name]
    data_generator = data_generator_class(**params)

    return data_generator

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
        self.ac2 = LogSoftmax()

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
