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
Constants for project.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import platform
import numpy as np

DL_FRAMEWORKS = np.array(['caffe', 'tensorflow', 'pytorch', 'keras', 'caffe2'])
DL_FRAMEWORK = None
GPU_CORE_ID = 0

CNN_FEATURE_SIZES = np.array([2048, 2048, 1000, 1024, 1000, 2048, 2048])
CNN_FEATURE_TYPES = np.array(['fc6', 'fc7', 'fc1000', 'fc1024', 'fc365', 'prob', 'pool5', 'fc8a', 'res3b7', 'res4b35', 'res5c'])
CNN_MODEL_TYPES = np.array(['resnet152', 'googlenet1k', 'vgg16', 'places365-resnet152', 'places365-vgg', 'googlenet13k'])
RESIZE_TYPES = np.array(['resize', 'resize_crop', 'resize_crop_scaled', 'resize_keep_aspect_ratio_padded'])
ROOT_PATH_TYPES = np.array(['data', 'project'])
TRAIN_SCHEMES = np.array(['ete', 'tco'])
MODEL_CLASSIFICATION_TYPES = np.array(['ml', 'sl'])
MODEL_MULTISCALE_TYPES = np.array(['dl', 'ks'])
SOLVER_NAMES = np.array(['adam', 'sgd'])
DATASET_NAMES = np.array(['charades', 'kinetics400', 'breakfast_actions', 'you_cook_2', 'multi_thumos'])
DATA_ROOT_PATH = './data'
PROJECT_ROOT_PATH = '../'
MACHINE_NAME = platform.node()
