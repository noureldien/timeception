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

import os
import logging
import json
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import torchviz
import torchvision
import torchsummary

logger = logging.getLogger(__name__)

# region Helpers

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

def summary(model, input_size, batch_size=-1, device="cuda"):
    """
    Custom summary function, to print the custom name of module, instead of the assigned layer name.
    :param model:
    :param input_size:
    :param batch_size:
    :param device:
    :return:
    """

    # this has to be imported here, not to create import-loop between "nets.layers_pytorch" and "core.pytorch_utils"
    from nets.layers_pytorch import DepthwiseConv1DLayer

    def register_hook(module):

        def hook(module, input, output):

            # old code
            # class_name = str(module.__class__).split(".")[-1].split("'")[0]
            # m_key = "%s-%i" % (class_name, module_idx + 1)

            # don't consider this layer
            if type(module) == DepthwiseConv1DLayer:
                return

            # new code
            if hasattr(module, '_name'):
                m_key = str(module._name)
            else:
                module_idx = len(summary)
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                m_key = "%s-%i" % (class_name, module_idx + 1)

            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(layer, str(summary[layer]["output_shape"]), "{0:,}".format(summary[layer]["nb_params"]), )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary

# endregion

# region Classes

class ModelSaver():
    def __init__(self, model, dataset_name, model_name):
        self.model = model
        self.model_name = model_name

        model_root_path = './data/%s/models' % (dataset_name)
        assert os.path.exists(model_root_path)

        model_root_path = './data/%s/models/%s' % (dataset_name, model_name)
        if not os.path.exists(model_root_path):
            os.mkdir(model_root_path)

        self.model_root_path = model_root_path

    def save(self, idx_epoch):
        """
        Save the model.
        """
        epoch_num = idx_epoch + 1
        model_root_path = self.model_root_path
        model_state_path = str('%s/%03d.pt' % (model_root_path, epoch_num))

        # save model state using pytorch
        model_state = self.model.state_dict()
        torch.save(model_state, model_state_path)


# endregion
