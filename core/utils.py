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
Helper functions for many things. Also, some needed classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import time
import h5py
import yaml
import numpy as np
import pickle as pkl
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing, manifold
import scipy.io as sio

import os
import json
import natsort
import random
from multiprocessing.dummy import Pool

from core import const

logger = logging.getLogger(__name__)

# region Load and Dump

def pkl_load(path):
    with open(path, 'r') as f:
        data = pkl.load(f)
    return data

def txt_load(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    lines = np.array(lines)
    return lines

def byte_load(path):
    with open(path, 'rb') as f:
        data = f.read()
    return data

def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data

def yaml_load(file_path):
    with open(file_path, 'r') as f:
        data = yaml.load(f)
        data = AttrDict(data)

    data = convert_dict_to_attrdict(data)
    return data

def h5_load(path, dataset_name='data'):
    h5_file = h5py.File(path, 'r')
    data = h5_file[dataset_name].value
    h5_file.close()
    return data

def h5_load_multi(path, dataset_names):
    h5_file = h5py.File(path, 'r')
    data = [h5_file[name].value for name in dataset_names]
    h5_file.close()
    return data

def txt_dump(data, path):
    l = len(data) - 1
    with open(path, 'w') as f:
        for i, k in enumerate(data):
            if i < l:
                k = ('%s\n' % k)
            else:
                k = ('%s' % k)
            f.writelines(k)

def byte_dump(data, path):
    with open(path, 'wb') as f:
        f.write(data)

def pkl_dump(data, path, is_highest=True):
    with open(path, 'w') as f:
        if not is_highest:
            pkl.dump(data, f)
        else:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

def json_dump(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def h5_dump(data, path, dataset_name='data'):
    h5_file = h5py.File(path, 'w')
    h5_file.create_dataset(dataset_name, data=data, dtype=data.dtype)
    h5_file.close()

def h5_dump_multi(data, dataset_names, path):
    h5_file = h5py.File(path, 'w')
    n_items = len(data)
    for i in range(n_items):
        item_data = data[i]
        item_name = dataset_names[i]
        h5_file.create_dataset(item_name, data=item_data, dtype=item_data.dtype)
    h5_file.close()

def csv_load(path, sep=',', header='infer'):
    df = pd.read_csv(path, sep=sep, header=header)
    data = df.values
    return data

def mat_load(path, m_dict=None):
    """
    Load mat files.
    :param path:
    :return:
    """
    if m_dict is None:
        data = sio.loadmat(path)
    else:
        data = sio.loadmat(path, m_dict)

    return data

# endregion

# region File/Folder Names/Pathes

def file_names(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).next()[2]

    if is_nat_sort:
        names = natsort.natsorted(names)

    return names

def file_pathes(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).next()[2]

    if is_nat_sort:
        names = natsort.natsorted(names)

    pathes = ['%s/%s' % (path, n) for n in names]
    return pathes

def folder_names(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).next()[1]

    if is_nat_sort:
        names = natsort.natsorted(names)

    return names

def folder_pathes(path, is_nat_sort=False):
    if not os.path.exists(path):
        exp_msg = 'Sorry, folder path does not exist: %s' % (path)
        raise Exception(exp_msg)

    names = os.walk(path).next()[1]

    if is_nat_sort:
        names = natsort.natsorted(names)

    pathes = ['%s/%s' % (path, n) for n in names]
    return pathes

# endregion

# region Normalization

def normalize_mean_std(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x -= mean
    x /= std
    return x

def normalize_mean(x):
    mean = np.mean(x, axis=0)
    x /= mean
    return x

def normalize_sum(x):
    sum = np.sum(x, axis=1)
    x = np.array([x_i / sum_i for x_i, sum_i in zip(x, sum)])
    return x

def normalize_l2(x):
    return preprocessing.normalize(x)

def normalize_l1(x):
    return preprocessing.normalize(x, norm='l1')

def normalize_range_0_to_1(x):
    x = np.add(x, -x.min())
    x = np.divide(x, x.max())
    return x

# endregion

# region Array Helpers

def array_to_text(a, separator=', '):
    text = separator.join([str(s) for s in a])
    return text

def get_size_in_kb(size):
    size /= float(1024)
    return size

def get_size_in_mb(size):
    size /= float(1024 * 1024)
    return size

def get_size_in_gb(size):
    size /= float(1024 * 1024 * 1024)
    return size

def get_array_memory_size(a):
    if type(a) is not np.ndarray:
        raise Exception('Sorry, input is not numpy array!')

    dtype = a.dtype
    if dtype == np.float16:
        n_bytes = 2
    elif dtype == np.float32:
        n_bytes = 4
    else:
        raise Exception('Sorry, unsupported dtype:', dtype)

    s = a.size
    size = s * n_bytes
    return size

def get_expected_memory_size(array_shape, array_dtype):
    dtype = array_dtype
    if dtype == np.float16:
        n_bytes = 2
    elif dtype == np.float32:
        n_bytes = 4
    else:
        raise Exception('Sorry, unsupported dtype:', dtype)

    s = 1
    for dim_size in array_shape:
        s *= dim_size

    size = s * n_bytes
    return size

def print_array(a):
    for item in a:
        print(item)

def print_array_joined(a):
    s = ', '.join([str(i) for i in a])
    print(s)

# endregion

# region Misc

def learn_manifold(manifold_type, feats, n_components=2):
    if manifold_type == 'tsne':
        feats_fitted = manifold.TSNE(n_components=n_components, random_state=0).fit_transform(feats)
    elif manifold_type == 'isomap':
        feats_fitted = manifold.Isomap(n_components=n_components).fit_transform(feats)
    elif manifold_type == 'mds':
        feats_fitted = manifold.MDS(n_components=n_components).fit_transform(feats)
    elif manifold_type == 'spectral':
        feats_fitted = manifold.SpectralEmbedding(n_components=n_components).fit_transform(feats)
    else:
        raise Exception('wrong maniford type!')

    # methods = ['standard', 'ltsa', 'hessian', 'modified']
    # feats_fitted = manifold.LocallyLinearEmbedding(n_components=n_components, method=methods[0]).fit_transform(pred)

    return feats_fitted

def debinarize_label(labels):
    debinarized = np.array([np.where(l == 1)[0][0] for l in labels])
    return debinarized

def timestamp():
    time_stamp = "{0:%y}.{0:%m}.{0:%d}-{0:%I}:{0:%M}:{0:%S}".format(datetime.now())
    return time_stamp

def remove_extension(name):
    name = name[:-4]
    return name

def get_file_extension(name):
    name = name.split('.')[-1]
    return name

def print_counter(num, total, freq=None):
    if freq is None:
        logger.info('... %d/%d' % (num, total))
    elif num % freq == 0:
        logger.info('... %d/%d' % (num, total))

def calc_num_batches(n_samples, batch_size):
    n_batch = int(n_samples / float(batch_size))
    n_batch = n_batch if n_samples % batch_size == 0 else n_batch + 1
    return n_batch

def convert_dict_to_attrdict(d):
    for k, v in d.iteritems():
        if isinstance(v, dict):
            v = convert_dict_to_attrdict(v)
            d[k] = v

    if isinstance(d, dict):
        d = AttrDict(d)

    return d

def get_model_feat_maps_info(model_type, feature_type):
    """
    Get feature map details according to model type and feature type.
    :param model_type:
    :param feature_type:
    :return:
    """

    if model_type in ['vgg', 'vgg_charades_rgb']:
        if feature_type == 'pool5':
            return 512, 7, 7
        elif feature_type == 'conv5_3':
            return 512, 14, 14
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    elif model_type in ['resnet152', 'resnet152_charades_rgb']:
        if feature_type == 'res4b35':
            return 1024, 14, 14
        elif feature_type == 'res5c':
            return 2048, 7, 7
        elif feature_type == 'pool5':
            return 2048, 1, 1
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    elif model_type in ['i3d_rgb', 'i3d_pytorch_charades_rgb', 'i3d_kinetics_keras', 'i3d_keras_kinetics_rgb']:
        if feature_type == 'mixed_5c':
            return 1024, 7, 7
        elif feature_type == 'mixed_4f':
            return 832, 7, 7
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    elif model_type in ['i3d_resnet_50_kinetics_rgb', 'i3d_resnet_101_kinetics_rgb']:
        if feature_type == 'pool5':
            return 2048, 7, 7
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    elif model_type in ['i3d_resnet101_charades_rgb']:
        if feature_type == 'res5_2':
            return 2048, 7, 7
        else:
            raise Exception('Sorry, unsupported feature type: %s' % (feature_type))
    else:
        raise Exception('Sorry, unsupported model type: %s' % (model_type))

# endregion

# region Classes

class Path(str):
    def __new__(self, relative_path, args=None, root_type=const.ROOT_PATH_TYPES[0]):
        assert root_type in const.ROOT_PATH_TYPES
        root_types = list(const.ROOT_PATH_TYPES)
        idx_root_type = root_types.index(root_type)

        root_paths = [const.DATA_ROOT_PATH, const.PROJECT_ROOT_PATH]
        root_path = root_paths[idx_root_type]

        relative_path = relative_path % args if args is not None else relative_path
        path = os.path.join(root_path, relative_path)

        self.__path = path
        return self.__path

    def __str__(self):
        return self.__path

    def __repr__(self):
        return self.__path

class DurationTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def duration(self, is_string=True):
        stop_time = time.time()
        durtation = stop_time - self.start_time
        if is_string:
            durtation = self.format_duration(durtation)
        return durtation

    def format_duration(self, duration):
        if duration < 60:
            return str(duration) + " sec"
        elif duration < (60 * 60):
            return str(duration / 60) + " min"
        else:
            return str(duration / (60 * 60)) + " hr"

class AttrDict(dict):
    """
    Subclass dict and define getter-setter. This behaves as both dict and obj.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

# endregion
