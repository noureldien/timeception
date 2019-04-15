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
ResNet-152 fine-tuned on Charades.
https://github.com/gsig/charades-algorithms/tree/master/pytorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import warnings
import os
import random
import sys
import time
import datetime
import math
import shutil
import random

import numpy as np
import cv2
import scipy.io
import h5py
from collections import OrderedDict

from core import const as c, utils
from core import image_utils

logger = logging.getLogger(__name__)

if c.DL_FRAMEWORK == 'tensorflow':
    import tensorflow as tf
elif c.DL_FRAMEWORK == 'caffe':
    import caffe
elif c.DL_FRAMEWORK == 'pytorch':
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    import torchvision.models as tmodels
    import importlib
elif c.DL_FRAMEWORK == 'keras':
    import tensorflow as tf
    import keras.backend as K

def get_resnet_152_charades_model():
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    import torchvision.models as tmodels
    import importlib
    import torch.utils.model_zoo as model_zoo

    root_path = c.DATA_ROOT_PATH
    model_arch = 'resnet152'
    model_checkpoint_path = '%s/Charades/baseline_models/resnet_rgb.pth.tar' % (root_path)

    # load model
    print("=> creating model '{}'".format(model_arch))
    model = tmodels.__dict__[model_arch](pretrained=False)
    cudnn.benchmark = True

    # load checkpoint
    checkpoint = torch.load(model_checkpoint_path)
    checkpoint = checkpoint['state_dict']

    # fix keys of state dict
    unwanted_keys = ['fc.weight', 'fc.bias']
    state_dict = OrderedDict()
    for k, v in checkpoint.iteritems():
        key = k.replace('module.', '')
        if key not in unwanted_keys:
            state_dict[key] = v

    # remove fc and avgpool layers
    layers = model._modules.items()
    layers = list(layers)[:-2]
    layers = OrderedDict(layers)
    model = nn.Sequential(layers)

    # load the dictionary
    model.load_state_dict(state_dict)

    # if parrallize the model
    # model = torch.nn.DataParallel(model).cuda()

    # make sure it's only for testing
    model.train(False)

    # convert to eval model
    model.eval()

    # convert to gpu model
    model.cuda()

    return model

def get_mean_std_for_resnet_152_pytorch_model():
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    return img_mean, img_std
