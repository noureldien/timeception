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
import os
import datetime
from optparse import OptionParser

import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, LeakyReLU, Dropout, Input, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization

from nets import timeception
from nets.layers_keras import MaxLayer
from core import utils, keras_utils, image_utils, config_utils, const, config, data_utils
from core.utils import Path as Pth

logger = logging.getLogger(__name__)

def test_tco():
    pass