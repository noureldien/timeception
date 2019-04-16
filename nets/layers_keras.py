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
Layers for keras.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from keras import backend as K
from keras.layers import Layer

import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers

logger = logging.getLogger(__name__)

# region Basic Layers

class SliceLayer(Layer):
    def __init__(self, name, **kwargs):
        self.name = name
        self.index = -1
        super(SliceLayer, self).__init__(**kwargs)

    def set_index(self, index):
        self.index = index

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2])
        return output_shape

    def call(self, input, mask=None):
        value = input[:, self.index, :]
        return value

class ReshapeLayer(Layer):
    def __init__(self, new_shape, **kwargs):
        self.new_shape = new_shape
        super(ReshapeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]] + list(self.new_shape)
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        output_shape = [-1] + list(self.new_shape)
        output_shape = tuple(output_shape)
        value = tf.reshape(input, output_shape)
        return value

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'new_shape': self.new_shape}
        base_config = super(ReshapeLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class TransposeLayer(Layer):
    def __init__(self, new_perm, **kwargs):
        self.new_perm = new_perm
        super(TransposeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == len(self.new_perm)

        output_shape = [input_shape[self.new_perm[idx]] for idx in range(len(input_shape))]
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        value = tf.transpose(input, self.new_perm)
        return value

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'new_perm': self.new_perm}
        base_config = super(TransposeLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(ExpandDimsLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]

        for axis in axes:
            output_shape.insert(axis, 1)

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):

        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        value = input

        for axis in axes:
            value = tf.expand_dims(value, axis)

        return value

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ExpandDimsLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(SqueezeLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if i not in axes:
                output_shape.append(input_shape[i])
        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        value = tf.squeeze(input, self.axis)
        return value

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class SqueezeAllLayer(Layer):
    def __init__(self, **kwargs):
        super(SqueezeAllLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SqueezeAllLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        output_shape = []
        for i in range(n_dims):
            dim_size = input_shape[i]
            if dim_size != 1:
                output_shape.append(dim_size)

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.squeeze(input)
        return tensor

    def get_config(self):
        config = {}
        base_config = super(SqueezeAllLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class MaxLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(MaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaxLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_max(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(MaxLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class SumLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(SumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SumLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_sum(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(SumLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class AverageLayer(Layer):
    def __init__(self, axis, is_keep_dim=False, **kwargs):
        self.axis = axis
        self.is_keep_dim = is_keep_dim
        super(AverageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AverageLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        n_dims = len(input_shape)
        axis_type = type(self.axis)
        axes = self.axis if (axis_type is list or axis_type is tuple) else [self.axis]
        output_shape = []
        for i in range(n_dims):
            if self.is_keep_dim:
                if i in axes:
                    output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            else:
                if i not in axes:
                    output_shape.append(input_shape[i])

        output_shape = tuple(output_shape)
        return output_shape

    def call(self, input, mask=None):
        tensor = tf.reduce_mean(input, axis=self.axis, keepdims=self.is_keep_dim)
        return tensor

    def get_config(self):
        config = {'axis': self.axis, 'is_keep_dim': self.is_keep_dim}
        base_config = super(AverageLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

# endregion

# region Timeception Layers

class NormalizationLayer(Layer):
    """
    Normalization layer, either l-1 or l-2 normalization.
    """

    def __init__(self, axis=None, **kwargs):
        self.axis = axis
        super(NormalizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NormalizationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = input_shape
        return output_dim

    def call(self, input, mask=None):
        tensor = tf.nn.l2_normalize(input, dim=self.axis)
        return tensor

    def get_config(self):
        config = {'axis', self.axis}
        base_config = super(NormalizationLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class DepthwiseConvOverTimeLayer(Layer):
    """
    Very similar to spatial conv, except we reshape the conv layer so that the last dimension
    is the temporal feature maps (i.e. spatial feature maps over time).
    Then we carry out conv on top of these temporal maps, then transpose back.
    """

    def __init__(self, channel_multiplier, depthwise_kernel_size, pointwise_kernel_size, **kwargs):
        self.channel_multiplier = channel_multiplier
        self.depthwise_kernel_size = depthwise_kernel_size
        self.pointwise_kernel_size = pointwise_kernel_size
        super(DepthwiseConvOverTimeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 20, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        initializer = contrib_layers.xavier_initializer()
        n_timesteps = input_shape[1]
        feat_map_side_dim = input_shape[2]
        n_spatial_maps = input_shape[4]

        self.n_timesteps = n_timesteps
        self.n_spatial_maps = n_spatial_maps
        self.feat_map_side_dim = feat_map_side_dim

        depthwise_weights_name1 = 'depthwise_weights1'
        depthwise_biases_name1 = 'depthwis_biases1'
        conv_weights_name = 'conv_weights'
        conv_biases_name = 'conv_biases'

        # we have n kernels (n=n_spatial_maps), each kernel is a 3-D matrix to spatially-convolve the i-th feature maps of of the m filters (m=n_timesteps) throughout the timesteps time
        depthwise_weights_shape = [self.depthwise_kernel_size, self.depthwise_kernel_size, n_spatial_maps, self.channel_multiplier]
        depthwise_bias_shape = [n_spatial_maps * self.channel_multiplier]

        # 1x1 convolution kernel
        conv_weights_shape = [n_spatial_maps, self.pointwise_kernel_size, self.pointwise_kernel_size, self.channel_multiplier, self.channel_multiplier]
        conv_bias_shape = [n_spatial_maps, self.channel_multiplier]

        with tf.variable_scope(self.name) as scope:
            self.depthwise_weights1 = tf.get_variable(depthwise_weights_name1, shape=depthwise_weights_shape, initializer=initializer)
            self.depthwise_bias1 = tf.get_variable(depthwise_biases_name1, shape=depthwise_bias_shape, initializer=tf.constant_initializer(0.1))
            self.conv_weights = tf.get_variable(conv_weights_name, shape=conv_weights_shape, initializer=initializer)
            self.conv_biases = tf.get_variable(conv_biases_name, shape=conv_bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.depthwise_weights1, self.depthwise_bias1, self.conv_weights, self.conv_biases]

        super(DepthwiseConvOverTimeLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.channel_multiplier, self.feat_map_side_dim, self.feat_map_side_dim, self.n_spatial_maps)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 20, 7, 7, 1024)

        # reshape for the sake of depthwise conv
        tensor = tf.reshape(input, (-1, self.feat_map_side_dim, self.feat_map_side_dim, self.n_spatial_maps))  # (None * 20, 7, 7, 1024)

        # depthwise conv with channel_multiplier (let's say channel multiplier is 3)
        tensor = tf.nn.depthwise_conv2d(tensor, self.depthwise_weights1, strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC')  # (None * 20, 7, 7, 3, 1024)
        tensor = tf.nn.bias_add(tensor, self.depthwise_bias1)

        # reshape to get channels over time is a separate dimension/axis
        tensor = tf.reshape(tensor, (-1, self.n_timesteps, self.feat_map_side_dim, self.feat_map_side_dim, self.channel_multiplier, self.n_spatial_maps))  # (None, 7, 7, 3, 1024)

        # max over time for each depthwise feature
        tensor = tf.reduce_max(tensor, axis=1)  # (None, 7, 7, 3, 1024)

        # apply 1x1 convolution on the temporal channels. Notice that we apply conv layer separately for each temporal channels
        # (i.e. channel_multiplications) originated from the same spatial channel (i.e. the 512 channels in the input)
        t_conv_list = []
        for i in range(self.n_spatial_maps):
            t = tensor[:, :, :, :, i]  # (None, 7, 7, 3)
            w = self.conv_weights[i]
            b = self.conv_biases[i]
            t_conv = tf.nn.conv2d(t, w, strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC')  # (None, 7, 7, 3)
            t_conv = tf.nn.bias_add(t_conv, b)
            t_conv = tf.expand_dims(t_conv, axis=4)  # (None, 7, 7, 3, 1)
            t_conv_list.append(t_conv)

        tensor = tf.concat(t_conv_list, axis=4)  # (None, 7, 7, 3, 1024)

        # transpose to get final shape
        tensor = tf.transpose(tensor, (0, 3, 1, 2, 4))  # (None, 3, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'channel_multiplier': self.channel_multiplier, 'depthwise_kernel_size': self.depthwise_kernel_size, 'pointwise_kernel_size': self.pointwise_kernel_size}
        base_config = super(DepthwiseConvOverTimeLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class GroupedDenseLayer(Layer):
    """
    Implementation of grouped dense layer. For a given n input units, we divide them into groups k groups,
    each group thus contain n/k units. Then, for each group, we apply dense layer which output l units. Thus,
    the total number of output units from this layer is l * k.
    """

    def __init__(self, n_groups, n_units_out_group, **kwargs):
        self.n_groups = n_groups
        self.n_units_out_group = n_units_out_group

        super(GroupedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 1024)
        :param input_shape:
        :return:
        """

        n_units_in = input_shape[1]
        n_units_out = self.n_units_out_group * self.n_groups

        self.n_units_out = n_units_out
        self.n_units_in = n_units_in

        self.n_units_in_group = int(n_units_in / self.n_groups)

        assert n_units_in % self.n_groups == 0

        initializer = contrib_layers.xavier_initializer()

        weight_shape = [self.n_groups, self.n_units_in_group, self.n_units_out_group]
        bias_shape = [self.n_groups, self.n_units_out_group]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable('conv_weights', shape=weight_shape, initializer=initializer)
            self.conv_biases = tf.get_variable('conv_biases', shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(GroupedDenseLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_units_out)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 1024)

        # split into groups
        tensor = tf.reshape(input, (-1, self.n_groups, self.n_units_in_group))  # (None, 4, 128)

        # transpose to be ready for matmul
        tensor = tf.transpose(tensor, (1, 0, 2))  # (4, None, 128)

        # apply weight and bias
        tensor = tf.matmul(tensor, self.conv_weights)  # (4, None, 128)
        tensor = tf.transpose(tensor, (1, 0, 2))  # (None, 4, 128)
        tensor = tf.add(tensor, self.conv_biases)

        # reshape to the final desired shape
        tensor = tf.reshape(tensor, (-1, self.n_units_out))  # (None, 4*128)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'n_units_out': self.n_units_out}
        base_config = super(GroupedDenseLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class DepthwiseDenseLayer(Layer):
    """
    MLP for the temporal dimension. Ignore the side dimension (i.e. ignore width and height), and the channel dimension.
    """

    def __init__(self, n_units_out, **kwargs):
        self.n_timesteps_out = n_units_out

        super(DepthwiseDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 10, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        assert len(input_shape) == 5

        _, self.n_timesteps_in, self.side_dim1, self.side_dim2, self.n_channels = input_shape

        initializer = contrib_layers.xavier_initializer()

        weight_shape = [self.n_channels, self.n_timesteps_in, self.n_timesteps_out]
        bias_shape = [self.n_channels, 1, self.n_timesteps_out]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable('dense_weights', shape=weight_shape, initializer=initializer)
            self.conv_biases = tf.get_variable('dense_biases', shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(DepthwiseDenseLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_timesteps_out, self.side_dim1, self.side_dim2, self.n_channels)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 10, 7, 7, 1024)

        # hide the side_dim with the batch_size
        tensor = tf.transpose(input, (4, 0, 2, 3, 1))  # (1024, None, 7, 7, 10)
        tensor = tf.reshape(tensor, (self.n_channels, -1, self.n_timesteps_in))

        # tensor: (1024, None, 10)
        # weight: (512, 10, 4)
        # bias:  (1024, None, 4)

        # apply weight and bias
        tensor = tf.matmul(tensor, self.conv_weights)  # (1024, None, 4)
        tensor = tf.add(tensor, self.conv_biases)  # (1024, None, 4)

        # reshape and transpose to get the desired output
        tensor = tf.reshape(tensor, (self.n_channels, -1, self.side_dim1, self.side_dim2, self.n_timesteps_out))  # (1024, None, 7, 7, 4)
        tensor = tf.transpose(tensor, (1, 4, 2, 3, 0))  # (None, 4, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'n_units_out': self.n_timesteps_out}
        base_config = super(DepthwiseDenseLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class DepthwiseConv1DLayer(Layer):
    """
    Expects a tensor of 5D (Batch_Size, Temporal_Dimension, Width, Length, Channel_Dimension)
    Applies a local 1*1*k Conv1D on each separate channel of the input, and along the temporal dimension
    Returns a 5D tensor.
    """

    def __init__(self, kernel_size, dilation_rate, padding, **kwargs):
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding
        super(DepthwiseConv1DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 20, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        assert len(input_shape) == 5

        initializer = contrib_layers.xavier_initializer()

        _, n_timesteps, feat_map_side_dim1, feat_map_side_dim2, n_spatial_maps = input_shape
        self.n_timesteps = n_timesteps
        self.n_maps = n_spatial_maps
        self.side_dim1 = feat_map_side_dim1
        self.side_dim2 = feat_map_side_dim2

        weights_name = 'depthwise_conv_1d_weights'
        biases_name = 'depthwise_conv1d_biases'

        # 1x1 convolution kernel
        weights_shape = [self.kernel_size, 1, n_spatial_maps, 1]
        bias_shape = [n_spatial_maps, ]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable(weights_name, shape=weights_shape, initializer=initializer)
            self.conv_biases = tf.get_variable(biases_name, shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(DepthwiseConv1DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_timesteps, self.side_dim1, self.side_dim2, self.n_maps)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 20, 7, 7, 1024)

        # transpose and reshape to hide the spatial dimension, only expose the temporal dimension for depthwise conv
        tensor = tf.transpose(input, (0, 2, 3, 1, 4))  # (None, 7, 7, 20, 1, 1024)
        tensor = tf.reshape(tensor, (-1, self.n_timesteps, 1, self.n_maps))  # (None*7*7, 20, 1, 1024)

        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = tf.nn.depthwise_conv2d(tensor, self.conv_weights, strides=(1, 1, 1, 1), rate=(self.dilation_rate, self.dilation_rate), padding='SAME', data_format='NHWC')  # (None*7*7, 20, 1, 1024)
        tensor = tf.nn.bias_add(tensor, self.conv_biases)  # (None*7*7, 20, 1, 1024)

        # reshape to get the spatial dimensions
        tensor = tf.reshape(tensor, (-1, self.side_dim1, self.side_dim2, self.n_timesteps, self.n_maps))  # (None, 7, 7, 20, 1024)

        # finally, transpose to get the desired output shape
        tensor = tf.transpose(tensor, (0, 3, 1, 2, 4))  # (None, 20, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'kernel_size': self.kernel_size, 'dilation_rate': self.dilation_rate, 'padding': self.padding}
        base_config = super(DepthwiseConv1DLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class DepthwiseConv2DLayer(Layer):
    """
    Expects a tensor of 4D (Batch_Size, Width, Length, Channel_Dimension)
    Applies a local k*k Conv2D on each separate channel of the input, and along the temporal dimension
    Returns a 4D tensor.
    """

    def __init__(self, kernel_size, padding, **kwargs):
        self.kernel_size = kernel_size
        self.padding = padding
        super(DepthwiseConv2DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        assert len(input_shape) == 4

        initializer = contrib_layers.xavier_initializer()

        _, feat_map_side_dim, _, n_spatial_maps = input_shape
        self.n_maps = n_spatial_maps
        self.side_dim = feat_map_side_dim

        weights_name = 'depthwise_conv_1d_weights'
        biases_name = 'depthwise_conv1d_biases'

        # kxk convolution kernel
        weights_shape = [self.kernel_size, self.kernel_size, n_spatial_maps, 1]
        bias_shape = [n_spatial_maps, ]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable(weights_name, shape=weights_shape, initializer=initializer)
            self.conv_biases = tf.get_variable(biases_name, shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(DepthwiseConv2DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.side_dim, self.side_dim, self.n_maps)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 7, 7, 1024)

        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = tf.nn.depthwise_conv2d(input, self.conv_weights, strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC')  # (None, 7, 7, 1024)
        tensor = tf.nn.bias_add(tensor, self.conv_biases)  # (None, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'kernel_size': self.kernel_size, 'padding': self.padding}
        base_config = super(DepthwiseConv2DLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class DepthwiseConv3DLayer(Layer):
    """
    Expects a tensor of 4D (Batch_Size, Width, Length, Channel_Dimension)
    Applies a local k*k Conv2D on each separate channel of the input, and along the temporal dimension
    Returns a 4D tensor.
    """

    def __init__(self, kernel_size, padding, **kwargs):
        self.kernel_size = kernel_size
        self.padding = padding
        super(DepthwiseConv3DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 64, 7, 7, 1024)
        :param input_shape:
        :return:
        """

        assert len(input_shape) == 5

        initializer = contrib_layers.xavier_initializer()

        _, n_timesteps, side_dim1, side_dim2, n_spatial_maps = input_shape
        self.n_maps = n_spatial_maps
        self.n_timesteps = n_timesteps
        self.side_dim_in_1 = side_dim1
        self.side_dim_in_2 = side_dim2

        self.side_dim_out_1 = self.__cal_side_dim(self.side_dim_in_1)
        self.side_dim_out_2 = self.__cal_side_dim(self.side_dim_in_2)

        weights_name = 'depthwise_conv_1d_weights'
        biases_name = 'depthwise_conv1d_biases'

        # kxk convolution kernel
        weights_shape = [self.kernel_size, self.kernel_size, n_spatial_maps, 1]
        bias_shape = [n_spatial_maps, ]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable(weights_name, shape=weights_shape, initializer=initializer)
            self.conv_biases = tf.get_variable(biases_name, shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(DepthwiseConv3DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_timesteps, self.side_dim_out_1, self.side_dim_out_2, self.n_maps)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, 64, 7, 7, 1024)

        # hide temporal dimension
        tensor = tf.reshape(input, (-1, self.side_dim_in_1, self.side_dim_in_2, self.n_maps))  # (None, 7, 7, 1024)

        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = tf.nn.depthwise_conv2d(tensor, self.conv_weights, strides=(1, 1, 1, 1), padding=self.padding, data_format='NHWC')  # (None, 7, 7, 1024)
        tensor = tf.nn.bias_add(tensor, self.conv_biases)  # (None, 7, 7, 1024)

        # restore temporal dimension
        tensor = tf.reshape(tensor, (-1, self.n_timesteps, self.side_dim_out_1, self.side_dim_out_2, self.n_maps))  # (None, 64, 7, 7, 1024)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'kernel_size': self.kernel_size, 'padding': self.padding}
        base_config = super(DepthwiseConv3DLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

    def __cal_side_dim(self, side_dim):
        padding = self.padding
        assert padding in ['VALID', 'valid', 'SAME', 'same']

        if self.padding in ['VALID', 'valid']:
            side_dim = side_dim - self.kernel_size + 1

        return side_dim

class GroupedConv3DLayer(Layer):
    """
    Implementation of grouped dense layer. For a given n input units, we divide them into groups k groups,
    each group thus contain n/k units. Then, for each group, we apply dense layer which output l units. Thus,
    the total number of output units from this layer is l * k.
    """

    def __init__(self, n_groups, n_units_out_group, **kwargs):
        self.n_groups = n_groups
        self.n_units_out_group = n_units_out_group

        super(GroupedConv3DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Input shape is (None, 1024)
        :param input_shape:
        :return:
        """

        rank = len(input_shape)
        assert rank == 5

        self.n_timesteps = input_shape[1]
        self.side_dim1 = input_shape[2]
        self.side_dim2 = input_shape[3]
        n_units_in = input_shape[4]
        n_units_out = self.n_units_out_group * self.n_groups

        self.n_units_out = n_units_out
        self.n_units_in = n_units_in

        self.n_units_in_group = int(n_units_in / self.n_groups)

        assert n_units_in % self.n_groups == 0

        initializer = contrib_layers.xavier_initializer()

        weight_shape = [self.n_groups, self.n_units_in_group, self.n_units_out_group]
        bias_shape = [self.n_groups, self.n_units_out_group]

        with tf.variable_scope(self.name) as scope:
            self.conv_weights = tf.get_variable('conv_weights', shape=weight_shape, initializer=initializer)
            self.conv_biases = tf.get_variable('conv_biases', shape=bias_shape, initializer=tf.constant_initializer(0.1))

        self.trainable_weights = [self.conv_weights, self.conv_biases]

        super(GroupedConv3DLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_dim = (None, self.n_timesteps, self.side_dim1, self.side_dim2, self.n_units_out)
        return output_dim

    def call(self, input, mask=None):
        # inputs is of shape (None, T, 7, 7, 1024)

        # split into groups
        tensor = tf.reshape(input, (-1, self.n_groups, self.n_units_in_group))  # (None*T*7*7, 4, 128)

        # transpose to be ready for matmul
        tensor = tf.transpose(tensor, (1, 0, 2))  # (4, None*T*7*7, 128)

        # apply weight and bias
        tensor = tf.matmul(tensor, self.conv_weights)  # (4, None*T*7*7, 128)
        tensor = tf.transpose(tensor, (1, 0, 2))  # (None*T*7*7, 4, 128)
        tensor = tf.add(tensor, self.conv_biases)

        # reshape to the final desired shape
        tensor = tf.reshape(tensor, (-1, self.n_timesteps, self.side_dim1, self.side_dim2, self.n_units_out))  # (None, 4*128)

        return tensor

    def get_config(self):
        """
        For rebuilding models on load time.
        """
        config = {'n_units_out': self.n_units_out}
        base_config = super(GroupedConv3DLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class ChannelShuffleLayer(Layer):
    def __init__(self, n_groups, **kwargs):
        self.n_groups = n_groups
        super(ChannelShuffleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.n_timesteps, self.side_dim1, self.side_dim2, self.n_channels = input_shape

        self.n_channels_per_group = int(self.n_channels / self.n_groups)
        assert self.n_channels_per_group * self.n_groups == self.n_channels

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape

    def call(self, input, mask=None):
        n_groups = self.n_groups
        n_channels = self.n_channels
        n_timesteps = self.n_timesteps
        side_dim1 = self.side_dim1
        side_dim2 = self.side_dim2
        n_channels_per_group = self.n_channels_per_group

        tensor = tf.reshape(input, (-1, n_timesteps, side_dim1, side_dim2, n_groups, n_channels_per_group))
        tensor = tf.transpose(tensor, (0, 1, 2, 3, 5, 4))
        tensor = tf.reshape(tensor, (-1, n_timesteps, side_dim1, side_dim2, n_channels))

        return tensor

    def get_config(self):
        config = {'n_groups': self.n_groups}
        base_config = super(ChannelShuffleLayer, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

# endregion
