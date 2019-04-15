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
Test Timeception models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import warnings
import random
import os
import sys
import shutil
import time
import datetime
import csv
import itertools
import math
import threading
import gc
import argparse
import natsort
import cv2
import glob
import numpy as np
from optparse import OptionParser
from sklearn.preprocessing import label_binarize

import tensorflow as tf
import keras.backend as K
import keras.layers
from keras import optimizers
from keras.layers import Dense, LeakyReLU, Dropout, Lambda, Conv2D, Activation, Conv3D, MaxPooling3D
from keras.layers import Input, merge, concatenate, GRU, Concatenate, DepthwiseConv2D, Multiply
from keras.optimizers import RMSprop, SGD, Adadelta, Adam, Adagrad, Adamax, Nadam
from keras.models import Sequential, Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import multi_gpu_utils
from keras.callbacks import LambdaCallback, Callback

from nets import timeception
from nets.keras_layers import DepthwiseConvOverTimeLayer, ReshapeLayer, TransposeLayer
from nets.keras_layers import GroupedDenseLayer, MaxLayer, AverageLayer, SumLayer
from nets.keras_layers import DepthwiseConv1DLayer, DepthwiseConv1DLayer, DepthwiseConv3DLayer, DepthwiseConv2DLayer
from core import utils, keras_utils, image_utils, config_utils, const, config, data_utils

logger = logging.getLogger(__name__)

def test_tco():
    pass