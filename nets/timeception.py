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
Definitio of Timeception as either keras layers or keras model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.layers import Concatenate, BatchNormalization, Activation, Lambda, MaxPooling3D, Conv3D

from nets.layers_keras import DepthwiseConv1DLayer, ChannelShuffleLayer

# region Timeception as Layers

def timeception_layers(tensor, n_layers=4, n_groups=8, is_dilated=True):
    input_shape = K.int_shape(tensor)
    assert len(input_shape) == 5

    expansion_factor = 1.25
    _, n_timesteps, side_dim, side_dim, n_channels_in = input_shape

    # how many layers of timeception
    for i in range(n_layers):
        layer_num = i + 1

        # get details about grouping
        n_channels_per_branch, n_channels_out = __get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)

        # temporal conv per group
        tensor = __grouped_convolutions(tensor, n_groups, n_channels_per_branch, is_dilated, layer_num)

        # downsample over time
        tensor = MaxPooling3D(pool_size=(2, 1, 1), name='maxpool_tc%d' % (layer_num))(tensor)
        n_channels_in = n_channels_out

    return tensor

def __grouped_convolutions(tensor_input, n_groups, n_channels_per_branch, is_dilated, layer_num):
    _, n_timesteps, side_dim1, side_dim2, n_channels_in = tensor_input.get_shape().as_list()
    assert n_channels_in % n_groups == 0
    n_branches = 5

    n_channels_per_group_in = int(n_channels_in / n_groups)
    n_channels_out = int(n_groups * n_branches * n_channels_per_branch)
    n_channels_per_group_out = int(n_channels_out / n_groups)

    assert n_channels_out % n_groups == 0

    # slice maps into groups
    layer_name = 'slice_groups_tc%d' % (layer_num)
    tensors = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)], name=layer_name)(tensor_input)

    # type of multi-scale kernels to use: either multi_kernel_sizes or multi_dilation_rates
    if is_dilated:
        kernel_sizes = (3, 3, 3)
        dilation_rates = (1, 2, 3)
    else:
        kernel_sizes = (3, 5, 7)
        dilation_rates = (1, 1, 1)

    # loop on groups
    t_outputs = []
    for idx_group in range(n_groups):
        group_num = idx_group + 1

        tensor = tensors[idx_group]
        tensor = __temporal_convolutional_block(tensor, n_channels_per_branch, kernel_sizes, dilation_rates, layer_num, group_num)
        t_outputs.append(tensor)

    # concatenate channels of groups
    tensor = Concatenate(axis=4, name='concat_tc%d' % (layer_num))(t_outputs)

    # activation
    tensor = Activation('relu', name='relu_tc%d' % (layer_num))(tensor)

    # shuffle channels
    tensor = ChannelShuffleLayer(n_groups, name='shuffle_tc%d' % (layer_num))(tensor)

    return tensor

def __temporal_convolutional_block(tensor, n_channels_per_branch, kernel_sizes, dilation_rates, layer_num, group_num):
    """
    Define 5 branches of convolutions that operate of channels of each group.
    """

    # branch 1: dimension reduction only and no temporal conv
    t_1 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name='conv_b1_g%d_tc%d' % (group_num, layer_num))(tensor)
    t_1 = BatchNormalization(name='bn_b1_g%d_tc%d' % (group_num, layer_num))(t_1)

    # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
    t_2 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name='conv_b2_g%d_tc%d' % (group_num, layer_num))(tensor)
    t_2 = DepthwiseConv1DLayer(kernel_sizes[0], dilation_rates[0], padding='same', name='convdw_b2_g%d_tc%d' % (group_num, layer_num))(t_2)
    t_2 = BatchNormalization(name='bn_b2_g%d_tc%d' % (group_num, layer_num))(t_2)

    # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
    t_3 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name='conv_b3_g%d_tc%d' % (group_num, layer_num))(tensor)
    t_3 = DepthwiseConv1DLayer(kernel_sizes[1], dilation_rates[1], padding='same', name='convdw_b3_g%d_tc%d' % (group_num, layer_num))(t_3)
    t_3 = BatchNormalization(name='bn_b3_g%d_tc%d' % (group_num, layer_num))(t_3)

    # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
    t_4 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name='conv_b4_g%d_tc%d' % (group_num, layer_num))(tensor)
    t_4 = DepthwiseConv1DLayer(kernel_sizes[2], dilation_rates[2], padding='same', name='convdw_b4_g%d_tc%d' % (group_num, layer_num))(t_4)
    t_4 = BatchNormalization(name='bn_b4_g%d_tc%d' % (group_num, layer_num))(t_4)

    # branch 5: dimension reduction followed by temporal max pooling
    t_5 = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name='conv_b5_g%d_tc%d' % (group_num, layer_num))(tensor)
    t_5 = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same', name='maxpool_b5_g%d_tc%d' % (group_num, layer_num))(t_5)
    t_5 = BatchNormalization(name='bn_b5_g%d_tc%d' % (group_num, layer_num))(t_5)

    # concatenate channels of branches
    tensor = Concatenate(axis=4, name='concat_g%d_tc%d' % (group_num, layer_num))([t_1, t_2, t_3, t_4, t_5])

    return tensor

def __get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in):
    n_branches = 5
    n_channels_per_branch = int(n_channels_in * expansion_factor / float(n_branches * n_groups))
    n_channels_per_branch = int(n_channels_per_branch)
    n_channels_out = int(n_channels_per_branch * n_groups * n_branches)
    n_channels_out = int(n_channels_out)
    return n_channels_per_branch, n_channels_out

# endregion

# region Timeception as Model

class Timeception(Model):
    """
    Timeception is defined as a keras model.
    """

    def __init__(self, n_channels_in, n_layers=4, n_groups=8, is_dilated=True, **kwargs):

        super(Timeception, self).__init__(**kwargs)

        expansion_factor = 1.25
        self.expansion_factor = expansion_factor

        self.n_channels_in = n_channels_in
        self.n_layers = n_layers
        self.n_groups = n_groups
        self.is_dilated = is_dilated

        self.__define_timeception_layers(n_channels_in, n_layers, n_groups, expansion_factor, is_dilated)

    def compute_output_shape(self, input_shape):
        n_layers = self.n_layers
        n_groups = self.n_groups
        expansion_factor = self.expansion_factor
        _, n_timesteps, side_dim_1, side_dim_2, n_channels_in = input_shape
        n_channels_out = n_channels_in

        for l in range(n_layers):
            n_timesteps = int(n_timesteps / 2.0)
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)
            n_channels_in = n_channels_out

        output_shape = (None, n_timesteps, side_dim_1, side_dim_2, n_channels_out)
        return output_shape

    def call(self, input, mask=None):

        n_layers = self.n_layers
        n_groups = self.n_groups
        expansion_factor = self.expansion_factor

        output = self.__call_timeception_layers(input, n_layers, n_groups, expansion_factor)

        return output

    def __define_timeception_layers(self, n_channels_in, n_layers, n_groups, expansion_factor, is_dilated):
        """
        Define layers inside the timeception layers.
        """

        # how many layers of timeception
        for i in range(n_layers):
            layer_num = i + 1

            # get details about grouping
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)

            # temporal conv per group
            self.__define_grouped_convolutions(n_channels_in, n_groups, n_channels_per_branch, is_dilated, layer_num)

            # downsample over time
            layer_name = 'maxpool_tc%d' % (layer_num)
            layer = MaxPooling3D(pool_size=(2, 1, 1), name=layer_name)
            setattr(self, layer_name, layer)

            n_channels_in = n_channels_out

    def __define_grouped_convolutions(self, n_channels_in, n_groups, n_channels_per_branch, is_dilated, layer_num):
        """
        Define layers inside grouped convolutional block.
        :return:
        """

        n_branches = 5
        n_channels_per_group_in = int(n_channels_in / n_groups)
        n_channels_out = int(n_groups * n_branches * n_channels_per_branch)
        n_channels_per_group_out = int(n_channels_out / n_groups)

        assert n_channels_in % n_groups == 0
        assert n_channels_out % n_groups == 0

        # slice maps into groups
        layer_name = 'slice_groups_tc%d' % (layer_num)
        layer = Lambda(lambda x: [x[:, :, :, :, i * n_channels_per_group_in:(i + 1) * n_channels_per_group_in] for i in range(n_groups)], name=layer_name)
        setattr(self, layer_name, layer)

        # type of multi-scale kernels to use: either multi_kernel_sizes or multi_dilation_rates
        if is_dilated:
            kernel_sizes = (3, 3, 3)
            dilation_rates = (1, 2, 3)
        else:
            kernel_sizes = (3, 5, 7)
            dilation_rates = (1, 1, 1)

        # loop on groups, and define convolutions in each group
        for idx_group in range(n_groups):
            group_num = idx_group + 1
            self.__define_temporal_convolutional_block(n_channels_per_branch, kernel_sizes, dilation_rates, layer_num, group_num)

        # concatenate channels of groups
        layer_name = 'concat_tc%d' % (layer_num)
        layer = Concatenate(axis=4, name=layer_name)
        setattr(self, layer_name, layer)

        # activation
        layer_name = 'relu_tc%d' % (layer_num)
        layer = Activation('relu', name=layer_name)
        setattr(self, layer_name, layer)

        # shuffle channels
        layer_name = 'shuffle_tc%d' % (layer_num)
        layer = ChannelShuffleLayer(n_groups, name=layer_name)
        setattr(self, layer_name, layer)

    def __define_temporal_convolutional_block(self, n_channels_per_branch, kernel_sizes, dilation_rates, layer_num, group_num):
        """
        Define 5 branches of convolutions that operate of channels of each group.
        """

        # branch 1: dimension reduction only and no temporal conv
        layer_name = 'conv_b1_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b1_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNormalization(name=layer_name)
        setattr(self, layer_name, layer)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        layer_name = 'conv_b2_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b2_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(kernel_sizes[0], dilation_rates[0], padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b2_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNormalization(name=layer_name)
        setattr(self, layer_name, layer)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        layer_name = 'conv_b3_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b3_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(kernel_sizes[1], dilation_rates[1], padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b3_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNormalization(name=layer_name)
        setattr(self, layer_name, layer)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        layer_name = 'conv_b4_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b4_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(kernel_sizes[2], dilation_rates[2], padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b4_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNormalization(name=layer_name)
        setattr(self, layer_name, layer)

        # branch 5: dimension reduction followed by temporal max pooling
        layer_name = 'conv_b5_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3D(n_channels_per_branch, kernel_size=(1, 1, 1), padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'maxpool_b5_g%d_tc%d' % (group_num, layer_num)
        layer = MaxPooling3D(pool_size=(2, 1, 1), strides=(1, 1, 1), padding='same', name=layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b5_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNormalization(name=layer_name)
        setattr(self, layer_name, layer)

        # concatenate channels of branches
        layer_name = 'concat_g%d_tc%d' % (group_num, layer_num)
        layer = Concatenate(axis=4, name=layer_name)
        setattr(self, layer_name, layer)

    def __call_timeception_layers(self, tensor, n_layers, n_groups, expansion_factor):
        input_shape = K.int_shape(tensor)
        assert len(input_shape) == 5

        _, n_timesteps, side_dim, side_dim, n_channels_in = input_shape

        # how many layers of timeception
        for i in range(n_layers):
            layer_num = i + 1

            # get details about grouping
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)

            # temporal conv per group
            tensor = self.__call_grouped_convolutions(tensor, n_groups, n_channels_per_branch, layer_num)

            # downsample over time
            tensor = getattr(self, 'maxpool_tc%d' % (layer_num))(tensor)
            n_channels_in = n_channels_out

        return tensor

    def __call_grouped_convolutions(self, tensor_input, n_groups, n_channels_per_branch, layer_num):

        # slice maps into groups
        tensors = getattr(self, 'slice_groups_tc%d' % (layer_num))(tensor_input)

        # loop on groups
        t_outputs = []
        for idx_group in range(n_groups):
            group_num = idx_group + 1
            tensor = tensors[idx_group]
            tensor = self.__call_temporal_convolutional_block(tensor, n_channels_per_branch, layer_num, group_num)
            t_outputs.append(tensor)

        # concatenate channels of groups
        tensor = getattr(self, 'concat_tc%d' % (layer_num))(t_outputs)
        # activation
        tensor = getattr(self, 'relu_tc%d' % (layer_num))(tensor)
        # shuffle channels
        tensor = getattr(self, 'shuffle_tc%d' % (layer_num))(tensor)

        return tensor

    def __call_temporal_convolutional_block(self, tensor, n_channels_per_branch, layer_num, group_num):
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
        t_5 = getattr(self, 'bn_b5_g%d_tc%d' % (group_num, layer_num))(t_5)

        # concatenate channels of branches
        tensor = getattr(self, 'concat_g%d_tc%d' % (group_num, layer_num))([t_1, t_2, t_3, t_4, t_5])

        return tensor

    def __get_n_channels_per_branch(self, n_groups, expansion_factor, n_channels_in):
        n_branches = 5
        n_channels_per_branch = int(n_channels_in * expansion_factor / float(n_branches * n_groups))
        n_channels_per_branch = int(n_channels_per_branch)
        n_channels_out = int(n_channels_per_branch * n_groups * n_branches)
        n_channels_out = int(n_channels_out)

        return n_channels_per_branch, n_channels_out

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'n_channels_in': self.n_channels_in, 'n_layers': self.n_layers, 'n_groups': self.n_groups, 'is_dilated': self.is_dilated}

        # TODO: Implement get_config
        # what is really needed here is that get_config should not only get configuration of Timeception as a module, but get all the
        # configuration of the layers inside the Timeception module. This is essential in serializing the Timeception module.
        # Currently, we can only save weights. To use them later, one should define the networking calling the python code, then
        # use model.load_weights()
        # base_config = super(Timeception, self).get_config()
        # config.update(base_config)

        return config

# endregion
