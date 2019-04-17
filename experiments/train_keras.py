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

import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LeakyReLU, Dropout, Input, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import Model

from nets import timeception
from nets.layers_keras import MaxLayer
from core import utils, keras_utils, image_utils, config_utils, const, config, data_utils
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
    save_callback = keras_utils.SaveCallback(dataset_name, model_name)

    # load model
    model = __define_timeception_model()
    logger.info(model.summary())

    # train the model
    model.fit_generator(epochs=n_epochs, generator=data_generator_tr, validation_data=data_generator_te, use_multiprocessing=True, workers=n_workers, callbacks=[save_callback], verbose=2)

    logger.info('--- finish time')
    logger.info(datetime.datetime.now())

def train_ete():
    """
    Train Timeception layers based on the given configurations.
    This train scheme is End-to-end (ETE).
    """

    model = __define_timeception_model()

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
    data_generator_class = data_utils.KERAS_DATA_GENERATORS_DICT[dataset_name]
    data_generator = data_generator_class(**params)

    return data_generator

def __define_timeception_model():
    """
    Define Timeception classifier.
    """

    # some configurations for the model
    classification_type = config.cfg.MODEL.CLASSIFICATION_TYPE
    solver_name = config.cfg.SOLVER.NAME
    solver_lr = config.cfg.SOLVER.LR
    adam_epsilon = config.cfg.SOLVER.ADAM_EPSILON
    n_tc_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS
    backbone_name = config.cfg.MODEL.BACKBONE_CNN
    feature_name = config.cfg.MODEL.BACKBONE_FEATURE
    n_tc_layers = config.cfg.MODEL.N_TC_LAYERS
    n_classes = config.cfg.MODEL.N_CLASSES
    is_dilated = config.cfg.MODEL.MULTISCALE_TYPE
    n_channels_in, channel_h, channel_w = utils.get_model_feat_maps_info(backbone_name, feature_name)
    n_groups = int(n_channels_in / 128.0)

    # optimizer and loss for either multi-label "ml" or single-label "sl" classification
    if classification_type == 'ml':
        loss = keras_utils.LOSSES[3]
        output_activation = keras_utils.ACTIVATIONS[2]
        metric_function = keras_utils.map_charades
    else:
        loss = keras_utils.LOSSES[0]
        output_activation = keras_utils.ACTIVATIONS[3]
        metric_function = keras_utils.METRICS[0]

    # define the optimizer
    optimizer = SGD(lr=0.01) if solver_name == 'sgd' else Adam(lr=solver_lr, epsilon=adam_epsilon)

    # input layer
    input_shape = (n_tc_timesteps, channel_h, channel_w, n_channels_in)  # (T, H, W, C)
    tensor_input = Input(shape=input_shape, name='input')  # (T, H, W, C)

    # define timeception layers, as a standalone module
    timeception_module = timeception.Timeception(n_channels_in, n_tc_layers, n_groups, is_dilated=is_dilated)
    tensor = timeception_module(tensor_input)  # (T, H, W, C)

    # but if you fancy, you can define timeception layers as a series of layers
    # tensor = timeception.timeception_layers(tensor_input, n_tc_layers, n_groups, is_dilated=is_dilated) # (T, H, W, C)

    # max-pool over space-time
    tensor = MaxLayer(axis=(1, 2, 3), name='maxpool_t_s')(tensor)

    # dense layers for classification
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(512)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.2)(tensor)
    tensor = Dropout(0.25)(tensor)
    tensor = Dense(n_classes)(tensor)
    tensor_output = Activation(output_activation)(tensor)

    # define the model
    model = Model(inputs=tensor_input, outputs=tensor_output)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric_function])

    return model

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
