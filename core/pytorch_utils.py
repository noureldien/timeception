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
Helper functions for pytorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchviz
import torchvision
import torchsummary

logger = logging.getLogger(__name__)

def save_model(model, path):
    model.save_state_dict(path)

def load_model(model, path):
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)

def padding1d(tensor, filter):
    it, = tensor.shape[2:]
    ft = filter

    pt = max(0, (it - 1) + (ft - 1) + 1 - it)
    oddt = (pt % it != 0)

    mode = str('constant')
    if any([oddt]):
        pad = [0, int(oddt)]
        tensor = F.pad(tensor, pad, mode=mode)

    padding = (pt // it,)
    return tensor, padding

def padding3d(tensor, filter, mode=str('constant')):
    """
    Input shape (BN, C, T, H, W)
    """

    it, ih, iw = tensor.shape[2:]
    ft, fh, fw = filter.shape

    pt = max(0, (it - 1) + (ft - 1) + 1 - it)
    ph = max(0, (ih - 1) + (fh - 1) + 1 - ih)
    pw = max(0, (iw - 1) + (fw - 1) + 1 - iw)

    oddt = (pt % 2 != 0)
    oddh = (ph % 2 != 0)
    oddw = (pw % 2 != 0)

    if any([oddt, oddh, oddw]):
        pad = [0, int(oddt), 0, int(oddh), 0, int(oddw)]
        tensor = F.pad(tensor, pad, mode=mode)

    padding = (pt // 2, ph // 2, pw // 2)
    tensor = F.conv3d(tensor, filter, padding=padding)

    return tensor

def calc_padding_1d(input_size, kernel_size, stride=1, dilation=1):
    """
    Calculate the padding.
    """

    # i = input
    # o = output
    # p = padding
    # k = kernel_size
    # s = stride
    # d = dilation
    # the equation is
    # o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1
    # give that we want i = o, then we solve the equation for p gives us

    i = input_size
    s = stride
    k = kernel_size
    d = dilation

    padding = 0.5 * (k - i + s * (i - 1) + (k - 1) * (d - 1))
    padding = int(padding)

    return padding
