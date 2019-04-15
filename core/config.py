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
Definition for all configuration options for training/testing Timeception model on various datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

from core.utils import AttrDict

logger = logging.getLogger(__name__)

__C = AttrDict()
cfg = __C

# region Misc

__C.DEBUG = False  # is debugging
__C.NUM_GPUS = 1  # how many gups to use
__C.LOG_PERIOD = 10  # log period
__C.DATASET_NAME = str('')  # name of dataset

# endregion

# region Model

__C.MODEL = AttrDict()
__C.MODEL.CLASSIFICATION_TYPE = str('')  # either multi-label 'ml' or single-label 'sl'
__C.MODEL.N_CLASSES = 157  # how many classes as output
__C.MODEL.N_CHAMNNEL_GROUPS = 8  # how many channel groups
__C.MODEL.N_TC_LAYERS = 4  # number of timeception layers
__C.MODEL.N_TC_TIMESTEPS = 64  # how mant timesteps expected as input to the timeception layers
__C.MODEL.N_INPUT_TIMESTEPS = 512  # how many timesteps (i.e. frames) expected as an input to the backbone CNN
__C.MODEL.NAME = str('')  # name suffex for the model to be trained
__C.MODEL.BACKBONE_CNN = str('')  # which backbone cnn is used
__C.MODEL.BACKBONE_FEATURE = str('')  # type of feature output from backbone cnn
__C.MODEL.MULTISCALE_TYPE = str('')  # use multi-scale by dilation rate "dl" or multi-scale by kernel-size "ks"

# endregion

# region Train

__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 64 # batch size for training
__C.TRAIN.N_EPOCHS = 500 # how many training epochs
__C.TRAIN.SCHEME = str('')  # either 'ete' (end-to-end) or tco ('timeception-only')
__C.TRAIN.N_WORKERS = 10 #

# endregion

# region Test

__C.TEST = AttrDict()
__C.TEST.BATCH_SIZE = 64
__C.TEST.N_SAMPLES = 10

# endregion

# region Solver

__C.SOLVER = AttrDict()
__C.SOLVER.NAME = str('adam')
__C.SOLVER.LR = 0.0001
__C.SOLVER.ADAM_EPSILON = 1e-4
__C.SOLVER.SGD_WEIGHT_DECAY = 0.0001
__C.SOLVER.SGD_MOMENTUM = 0.9
__C.SOLVER.SGD_NESTEROV = True

# endregion
