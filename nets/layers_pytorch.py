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
Layers for pytorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging

import torch
from torch.nn import Module, Conv2d, Conv1d
from torch.nn import functional as F

from core import pytorch_utils

logger = logging.getLogger(__name__)

# region Basic Layers

class ChannelShuffleLayer(Module):
    """
    Shuffle the channels across groups.
    """

    def __init__(self, n_channels, n_groups):
        super(ChannelShuffleLayer, self).__init__()

        n_channels_per_group = int(n_channels / n_groups)
        assert n_channels_per_group * n_groups == n_channels

        self.n_channels_per_group = n_channels_per_group
        self.n_groups = n_groups

    def forward(self, input):
        """
        input shape (None, 1024, 20, 7, 7), or (BN, C, T, H, W)
        """

        input_shape = input.size()
        n_samples, n_channels, n_timesteps, side_dim1, side_dim2 = input_shape

        n_groups = self.n_groups
        n_channels_per_group = self.n_channels_per_group

        tensor = input.view(n_samples, n_groups, n_channels_per_group, n_timesteps, side_dim1, side_dim2)
        tensor = tensor.permute(0, 2, 1, 3, 4, 5)
        tensor = tensor.contiguous()
        tensor = tensor.view(n_samples, n_channels, n_timesteps, side_dim1, side_dim2)

        return tensor

# endregion

# region Timeception Layers

class DepthwiseConv1DLayer(Module):
    """
    Shuffle the channels across groups.
    """

    def __init__(self, input_shape, kernel_size, dilation, name):
        super(DepthwiseConv1DLayer, self).__init__()

        assert len(input_shape) == 5

        self.kernel_size = kernel_size
        self.dilation = dilation
        self._name = name

        n_channels = input_shape[1]
        n_timesteps = input_shape[2]

        # TODO: support using different dilation rates.
        padding = pytorch_utils.calc_padding_1d(n_timesteps, kernel_size)
        self.depthwise_conv1d = Conv1d(n_channels, n_channels, kernel_size, dilation=dilation, groups=n_channels, padding=padding)
        self.depthwise_conv1d._name = name

    def forward(self, input):
        """
        input shape (None, 1024, 20, 7, 7), or (BN, C, T, H, W)
        """

        input_shape = input.size()

        n, c, t, h, w = input_shape

        # transpose and reshape to hide the spatial dimension, only expose the temporal dimension for depthwise conv
        tensor = input.permute(0, 3, 4, 1, 2)  # (None, 7, 7, 1024, 20)
        tensor = tensor.contiguous()
        tensor = tensor.view(-1, c, t)  # (None*7*7, 1024, 20)

        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = self.depthwise_conv1d(tensor)  # (None*7*7, 1024, 20)

        # get timesteps after convolution
        t = tensor.size()[-1]

        # reshape to get the spatial dimensions
        tensor = tensor.view(n, h, w, c, t)  # (None, 7, 7, 1024, 20)

        # finally, transpose to get the desired output shape
        tensor = tensor.permute(0, 3, 4, 1, 2)  # (None, 1024, 20, 7, 7)

        return tensor

# endregion
