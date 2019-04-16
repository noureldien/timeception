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
Helper functions for keras.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import pydot
import logging
import numpy as np

import tensorflow as tf
from tensorflow.contrib import framework as tf_framework

import keras.backend as K
from keras.callbacks import Callback
from keras.utils import vis_utils
from keras.models import Sequential, model_from_json

from core import config_utils

logger = logging.getLogger(__name__)

# region Constants

EPS_VALUE = 1e-9
LOSSES = ['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy']
METRICS = ['accuracy', 'mean_squared_error', 'mean_absolute_error']
OPTIMIZERS = ['sgd', 'rmsprop', 'adam']
ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'softmax']

# endregion

# region Functions

def save_model_figure(model, file_path='/.model.eps'):
    vis_utils.plot_model(model, file_path, show_shapes=True, show_layer_names=True)

def load_model(json_path, weight_path, metrics=None, loss=None, optimizer=None, custom_objects=None, is_compile=True):
    with open(json_path, 'r') as f:
        model_json_string = json.load(f)
    model_json_dict = json.loads(model_json_string)
    model = model_from_json(model_json_string, custom_objects=custom_objects)
    model.load_weights(weight_path)

    if is_compile:
        if optimizer is None:
            optimizer = model_json_dict['optimizer']['name']

        if loss is None:
            loss = model_json_dict['loss']

        if metrics is None:
            model.compile(loss=loss, optimizer=optimizer)
        else:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def save_model(model, json_path, weight_path):
    model.save_weights(weight_path, overwrite=True)
    model_json = model.to_json()
    with open(json_path, 'w') as f:
        json.dump(model_json, f)

def layer_exist(model, layer_name):
    exist = False
    for layer in model.layers:
        if layer.name == layer_name:
            exist = True
            break

    return exist

def calc_num_batches(n_samples, batch_size):
    n_batch = int(n_samples / float(batch_size))
    n_batch = n_batch if n_samples % batch_size == 0 else n_batch + 1
    return n_batch

# endregion

# region Metrics

def map_charades(y_true, y_pred):
    """
    Returns mAP
    """
    m_aps = []

    tf_one = tf.constant(1, dtype=tf.float32)

    n_classes = y_pred.shape[1]
    for oc_i in range(n_classes):
        pred_row = y_pred[:, oc_i]
        sorted_idxs = tf_framework.argsort(-pred_row)
        true_row = y_true[:, oc_i]
        true_row = tf.map_fn(lambda i: true_row[i], sorted_idxs, dtype=np.float32)
        tp_poolean = tf.equal(true_row, tf_one)
        tp = tf.cast(tp_poolean, dtype=np.float32)
        fp = K.reverse(tp, axes=0)
        n_pos = tf.reduce_sum(tp)
        f_pcs = tf.cumsum(fp)
        t_pcs = tf.cumsum(tp)
        s = f_pcs + t_pcs

        s = tf.cast(s, tf.float32)
        t_pcs = tf.cast(t_pcs, tf.float32)
        tp_float = tf.cast(tp_poolean, np.float32)

        prec = t_pcs / s
        avg_prec = prec * tp_float

        n_pos = tf.cast(n_pos, tf.float32)
        avg_prec = avg_prec / n_pos
        avg_prec = tf.expand_dims(avg_prec, axis=0)
        m_aps.append(avg_prec)

    m_aps = K.concatenate(m_aps, axis=0)
    mAP = K.mean(m_aps)
    return mAP

# endregion

# region Callbacks

class SaveCallback(Callback):
    def __init__(self, dataset_name, model_name):
        self.model_name = model_name

        model_root_path = './data/%s/models' % (dataset_name)
        assert os.path.exists(model_root_path)

        model_root_path = './data/%s/models/%s' % (dataset_name, model_name)
        if not os.path.exists(model_root_path):
            os.mkdir(model_root_path)

        self.model_root_path = model_root_path

        super(SaveCallback, self).__init__()

    def on_epoch_end(self, idx_epoch, logs=None):
        """
        Save the model.
        """

        epoch_num = idx_epoch + 1
        self.__save(epoch_num)

    def __save(self, epoch_num):
        model_root_path = self.model_root_path
        model = self.model

        # hfpy accept only strings as a path
        model_json_path = str('%s/%03d.json' % (model_root_path, epoch_num))
        model_weight_path = str('%s/%03d.pkl' % (model_root_path, epoch_num))

        # save model definition as json, and save model weights
        model.save_weights(model_weight_path, overwrite=True)
        model_json = model.to_json()
        with open(model_json_path, 'w') as f:
            json.dump(model_json, f)

# endregion
