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
Definitio of Timeception as pytorch model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
import torch.nn
import torchvision
import torchviz
import torchsummary

from torch.nn import Module, Conv3d, BatchNorm3d, MaxPool3d, ReLU
from torch.nn import functional as F

from nets.layers_pytorch import ChannelShuffleLayer, DepthwiseConv1DLayer

# region Timeception as Module

class Timeception(Module):
    """
    Timeception is defined as a keras model.
    """

    def __init__(self, input_shape, n_layers=4, n_groups=8, is_dilated=True):

        super(Timeception, self).__init__()

        # TODO: Add support for multi-scale using dilation rates
        # current, for pytorch, we only support multi-scale using kernel sizes
        is_dilated = False

        expansion_factor = 1.25
        self.expansion_factor = expansion_factor
        self.n_layers = n_layers
        self.is_dilated = is_dilated
        self.n_groups = n_groups
        self.n_channels_out = None

        # convert it as a list
        input_shape = list(input_shape)

        # define timeception layers
        n_channels_out = self.__define_timeception_layers(input_shape, n_layers, n_groups, expansion_factor, is_dilated)

        # set the output channels
        self.n_channels_out = n_channels_out

    def forward(self, input):

        n_layers = self.n_layers
        n_groups = self.n_groups
        expansion_factor = self.expansion_factor

        output = self.__call_timeception_layers(input, n_layers, n_groups, expansion_factor)

        return output

    def __define_timeception_layers(self, input_shape, n_layers, n_groups, expansion_factor, is_dilated):
        """
        Define layers inside the timeception layers.
        """

        n_channels_in = input_shape[1]

        # how many layers of timeception
        for i in range(n_layers):
            layer_num = i + 1

            # get details about grouping
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)

            # temporal conv per group
            self.__define_grouped_convolutions(input_shape, n_groups, n_channels_per_branch, is_dilated, layer_num)

            # downsample over time
            layer_name = 'maxpool_tc%d' % (layer_num)
            layer = MaxPool3d(kernel_size=(2, 1, 1))
            layer._name = layer_name
            setattr(self, layer_name, layer)

            n_channels_in = n_channels_out
            input_shape[1] = n_channels_in

        return n_channels_in

    def __define_grouped_convolutions(self, input_shape, n_groups, n_channels_per_branch, is_dilated, layer_num):
        """
        Define layers inside grouped convolutional block.
        """

        n_channels_in = input_shape[1]

        n_branches = 5
        n_channels_per_group_in = int(n_channels_in / n_groups)
        n_channels_out = int(n_groups * n_branches * n_channels_per_branch)
        n_channels_per_group_out = int(n_channels_out / n_groups)

        assert n_channels_in % n_groups == 0
        assert n_channels_out % n_groups == 0

        # type of multi-scale kernels to use: either multi_kernel_sizes or multi_dilation_rates
        if is_dilated:
            kernel_sizes = (3, 3, 3)
            dilation_rates = (1, 2, 3)
        else:
            kernel_sizes = (3, 5, 7)
            dilation_rates = (1, 1, 1)

        input_shape_per_group = list(input_shape)
        input_shape_per_group[1] = n_channels_per_group_in

        # loop on groups, and define convolutions in each group
        for idx_group in range(n_groups):
            group_num = idx_group + 1
            self.__define_temporal_convolutional_block(input_shape_per_group, n_channels_per_branch, kernel_sizes, dilation_rates, layer_num, group_num)

        # activation
        layer_name = 'relu_tc%d' % (layer_num)
        layer = ReLU()
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # shuffle channels
        layer_name = 'shuffle_tc%d' % (layer_num)
        layer = ChannelShuffleLayer(n_channels_out, n_groups)
        layer._name = layer_name
        setattr(self, layer_name, layer)

    def __define_temporal_convolutional_block(self, input_shape, n_channels_per_branch_out, kernel_sizes, dilation_rates, layer_num, group_num):
        """
        Define 5 branches of convolutions that operate of channels of each group.
        """

        n_channels_in = input_shape[1]

        dw_input_shape = list(input_shape)
        dw_input_shape[1] = n_channels_per_branch_out

        # branch 1: dimension reduction only and no temporal conv
        layer_name = 'conv_b1_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'bn_b1_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        layer_name = 'conv_b2_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b2_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(dw_input_shape, kernel_sizes[0], dilation_rates[0], layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b2_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        layer_name = 'conv_b3_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b3_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(dw_input_shape, kernel_sizes[1], dilation_rates[1], layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b3_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        layer_name = 'conv_b4_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b4_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(dw_input_shape, kernel_sizes[2], dilation_rates[2], layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b4_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 5: dimension reduction followed by temporal max pooling
        layer_name = 'conv_b5_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'maxpool_b5_g%d_tc%d' % (group_num, layer_num)
        layer = MaxPool3d(kernel_size=(2, 1, 1), stride=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'padding_b5_g%d_tc%d' % (group_num, layer_num)
        layer = torch.nn.ReplicationPad3d((0, 0, 0, 0, 1, 0))  # left, right, top, bottom, front, back
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'bn_b5_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

    def __call_timeception_layers(self, tensor, n_layers, n_groups, expansion_factor):
        input_shape = tensor.size()
        n_channels_in = input_shape[1]

        # how many layers of timeception
        for i in range(n_layers):
            layer_num = i + 1

            # get details about grouping
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)

            # temporal conv per group
            tensor = self.__call_grouped_convolutions(tensor, n_groups, layer_num)

            # downsample over time
            tensor = getattr(self, 'maxpool_tc%d' % (layer_num))(tensor)
            n_channels_in = n_channels_out

        return tensor

    def __call_grouped_convolutions(self, tensor_input, n_groups, layer_num):

        n_channels_in = tensor_input.size()[1]
        n_channels_per_group_in = int(n_channels_in / n_groups)

        # loop on groups
        t_outputs = []
        for idx_group in range(n_groups):
            group_num = idx_group + 1

            # slice maps to get maps per group
            idx_start = idx_group * n_channels_per_group_in
            idx_end = (idx_group + 1) * n_channels_per_group_in
            tensor = tensor_input[:, idx_start:idx_end]

            tensor = self.__call_temporal_convolutional_block(tensor, layer_num, group_num)
            t_outputs.append(tensor)

        # concatenate channels of groups
        tensor = torch.cat(t_outputs, dim=1)
        # activation
        tensor = getattr(self, 'relu_tc%d' % (layer_num))(tensor)
        # shuffle channels
        tensor = getattr(self, 'shuffle_tc%d' % (layer_num))(tensor)

        return tensor

    def __call_temporal_convolutional_block(self, tensor, layer_num, group_num):
        """
        Feedforward for 5 branches of convolutions that operate of channels of each group.
        """

        # branch 1: dimension reduction only and no temporal conv
        t_1 = getattr(self, 'conv_b1_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_1 = getattr(self, 'bn_b1_g%d_tc%d' % (group_num, layer_num))(t_1)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        t_2 = getattr(self, 'conv_b2_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_2 = getattr(self, 'convdw_b2_g%d_tc%d' % (group_num, layer_num))(t_2)
        t_2 = getattr(self, 'bn_b2_g%d_tc%d' % (group_num, layer_num))(t_2)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        t_3 = getattr(self, 'conv_b3_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_3 = getattr(self, 'convdw_b3_g%d_tc%d' % (group_num, layer_num))(t_3)
        t_3 = getattr(self, 'bn_b3_g%d_tc%d' % (group_num, layer_num))(t_3)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        t_4 = getattr(self, 'conv_b4_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_4 = getattr(self, 'convdw_b4_g%d_tc%d' % (group_num, layer_num))(t_4)
        t_4 = getattr(self, 'bn_b4_g%d_tc%d' % (group_num, layer_num))(t_4)

        # branch 5: dimension reduction followed by temporal max pooling
        t_5 = getattr(self, 'conv_b5_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_5 = getattr(self, 'maxpool_b5_g%d_tc%d' % (group_num, layer_num))(t_5)
        t_5 = getattr(self, 'padding_b5_g%d_tc%d' % (group_num, layer_num))(t_5)
        t_5 = getattr(self, 'bn_b5_g%d_tc%d' % (group_num, layer_num))(t_5)

        # concatenate channels of branches
        tensors = (t_1, t_2, t_3, t_4, t_5)
        tensor = torch.cat(tensors, dim=1)

        return tensor

    def __get_n_channels_per_branch(self, n_groups, expansion_factor, n_channels_in):
        n_branches = 5
        n_channels_per_branch = int(n_channels_in * expansion_factor / float(n_branches * n_groups))
        n_channels_per_branch = int(n_channels_per_branch)
        n_channels_out = int(n_channels_per_branch * n_groups * n_branches)
        n_channels_out = int(n_channels_out)

        return n_channels_per_branch, n_channels_out

# endregion
