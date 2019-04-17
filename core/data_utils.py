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
Helpful functions and classes to deal with data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import random
import numpy as np
import pickle as pkl
from datetime import datetime
from multiprocessing.dummy import Pool

import keras.utils
import torch.utils.data
import torchvision

from core import utils, config
from core.utils import Path as Pth

logger = logging.getLogger(__name__)

# region Async File Loader

class AsyncLoaderVideoFeatures():
    """
    Load features for the video frames.
    """

    def __init__(self, feats_path, target, n_frames_per_video, batch_size, n_feat_maps, feat_map_side_dim, n_threads=10, annotation_dict=None):
        random.seed(101)
        np.random.seed(101)

        self.__feats_pathes = feats_path
        self.__n_frames_per_video = n_frames_per_video
        self.__n_feat_maps = n_feat_maps
        self.__feat_map_side_dim = feat_map_side_dim
        self.__annotation_dict = annotation_dict

        self.__batch_size = batch_size
        self.__y = target

        self.__is_busy = False
        self.__batch_features = None
        self.__batch_y = None
        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_feats_in_batch(self, batch_number):
        self.__is_busy = True

        idx_batch = batch_number - 1
        start_idx = idx_batch * self.__batch_size
        stop_idx = (idx_batch + 1) * self.__batch_size

        batch_feat_pathes = self.__feats_pathes[start_idx:stop_idx]
        batch_y = self.__y[start_idx:stop_idx]

        n_batch_feats = len(batch_feat_pathes)
        n_batch_y = len(batch_y)
        idxces = range(0, n_batch_feats)

        assert n_batch_feats == n_batch_y

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, batch_feat_pathes)]

        # set list of batch features before start reading
        batch_feats_shape = (n_batch_feats, self.__n_frames_per_video, self.__feat_map_side_dim, self.__feat_map_side_dim, self.__n_feat_maps)

        self.__batch_features = np.zeros(batch_feats_shape, dtype=np.float32)
        self.__batch_y = batch_y

        # start pool of threads
        self.__pool.map_async(self.__load_features, params, callback=self.__thread_pool_callback)

    def get_batch_data(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get features while threads are running!')
        else:
            return (self.__batch_features, self.__batch_y)

    def get_y(self):
        return self.__y

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __load_features(self, params):

        idx_video = params[0]
        feats_path = params[1]
        video_name = feats_path.split('/')[-1]

        try:
            # load feature from file
            feats = utils.pkl_load(feats_path)

            n_feats = len(feats)
            assert n_feats == self.__n_frames_per_video, 'Sorry, wrong number of frames, expected: %d, got: %d' % (self.__n_frames_per_video, n_feats)
            self.__batch_features[idx_video] = feats

        except Exception as exp:
            print('\nSorry, error in loading feature %s' % (feats_path))
            print(exp)

    def shuffle_data(self):
        """
        shuffle these data: self.__feats_pathes, self.__class_names, self.__y
        :return:
        """

        n_samples = len(self.__feats_pathes)

        idx = range(n_samples)
        np.random.shuffle(idx)
        self.__feats_pathes = self.__feats_pathes[idx]
        self.__y = self.__y[idx]

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion

# region Data Generators (Keras)

class DataGeneratorCharades(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size, n_classes, feature_dim, feature_name, is_training, is_shuffle=True):
        """
        Initialization
        """
        self.batch_size = batch_size
        self.is_training = is_training
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.feature_name = feature_name
        self.is_shuffle = is_shuffle
        self.dataset_name = 'charades'

        # load annotation
        root_path = './data/charades'
        annotation_path = '%s/annotation/video_annotation.pkl' % (root_path)
        if self.is_training:
            (video_names, y, _, _) = utils.pkl_load(annotation_path)
        else:
            (_, _, video_names, y) = utils.pkl_load(annotation_path)

        # convert relative to root pathes
        feats_path = np.array(['%s/%s/%s.pkl' % (root_path, feature_name, p) for p in video_names])

        n_samples = len(y)
        self.n_samples = n_samples
        self.n_batches = utils.calc_num_batches(n_samples, batch_size)
        self.feats_path = feats_path
        self.y = y

        # shuffle the data
        if self.is_shuffle:
            self.__shuffle()

    def __len__(self):
        """
        Denotes the number of batches per epoc
        """
        return self.n_batches

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        idx_start = index * self.batch_size
        idx_stop = (index + 1) * self.batch_size
        y = self.y[idx_start:idx_stop]
        feats_path = self.feats_path[idx_start:idx_stop]

        n_items = len(feats_path)
        x_shape = tuple([n_items] + list(self.feature_dim))
        x = np.zeros(x_shape, dtype=np.float32)

        # loop of feature pathes and load them
        for idx, p in enumerate(feats_path):
            x[idx] = utils.pkl_load(p)

        return x, y

    def on_epoch_end(self):
        """
        Shuffle after finishing the epoch.
        :return:
        """

        if self.is_shuffle:
            self.__shuffle()

    def __shuffle(self):

        idx = range(self.n_samples)
        np.random.shuffle(idx)
        self.feats_path = self.feats_path[idx]
        self.y = self.y[idx]

# endregion

# region Data Loaders (PyTorch)

class DatasetCharades(torch.utils.data.Dataset):
    def __init__(self, batch_size, n_classes, feature_dim, feature_name, is_training, is_shuffle=True):
        """
        Initialization
        """

        self.batch_size = batch_size
        self.is_training = is_training
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.feature_name = feature_name
        self.is_shuffle = is_shuffle
        self.dataset_name = 'charades'

        # load annotation
        root_path = './data/charades'
        annotation_path = '%s/annotation/video_annotation.pkl' % (root_path)
        if self.is_training:
            (video_names, y, _, _) = utils.pkl_load(annotation_path)
        else:
            (_, _, video_names, y) = utils.pkl_load(annotation_path)

        # in case of single label classification, debinarize the labels
        if config.cfg.MODEL.CLASSIFICATION_TYPE == 'sl':
            y = utils.debinarize_label(y)

        # in any case, make sure target is float
        y = y.astype(np.float32)

        # convert relative to root pathes
        feats_path = np.array(['%s/%s/%s.pkl' % (root_path, feature_name, p) for p in video_names])

        n_samples = len(y)
        self.n_samples = n_samples
        self.n_batches = utils.calc_num_batches(n_samples, batch_size)
        self.feats_path = feats_path
        self.y = y

        # shuffle the data
        if self.is_shuffle:
            self.__shuffle()

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        y = self.y[index]
        p = self.feats_path[index]
        x = utils.pkl_load(p)  # (T, H, W, C)

        # convert to channel last
        x = np.transpose(x, (3, 0, 1, 2))  # (T, H, W, C)

        return x, y

    def __len__(self):
        return self.n_samples

    def __shuffle(self):
        idx = range(self.n_samples)
        np.random.shuffle(idx)
        self.feats_path = self.feats_path[idx]
        self.y = self.y[idx]

# endregion

# region Constants

KERAS_DATA_GENERATORS_DICT = {'charades': DataGeneratorCharades}
PYTORCH_DATASETS_DICT = {'charades': DatasetCharades}

# endregion
